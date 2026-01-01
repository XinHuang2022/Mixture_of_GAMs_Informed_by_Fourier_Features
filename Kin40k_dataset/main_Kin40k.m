	clear

	%% Configure the virtual environment with correct Python version (3.10)
	pyenv("Version", "G:\Research\Projects\Post_Graduate\Explainable_Machine_Learning\Random_Feature_Models\codes_backup\pygam_virt_env\Scripts\python.exe", ...
		  "ExecutionMode", "OutOfProcess");

	addpath(genpath('utilities'));

	%% -----------------------------
	% 1) Load raw data (UCI format)
	% -----------------------------
	data_file = fullfile(pwd, 'data', 'data.csv');
	test_mask_file = fullfile(pwd, 'data', 'test_mask.csv');

	% Load
	data_all = readmatrix(data_file);
	% Load mask as matrix
	mask_matrix = readmatrix(test_mask_file);   % 40000 x 10
	% Use column 1 as the test set
	mask_test = logical(mask_matrix(:,1));  % 40000 x 1 logical

	% Split into predictors and targets
	X_all = data_all(:, 1:end-1);
	y_all = data_all(:, end);

	% Apply test mask
	X_train = X_all(~mask_test, :);
	y_train = y_all(~mask_test);
	X_test  = X_all(mask_test, :);
	y_test  = y_all(mask_test);

	fprintf("Train size: %d, Test size: %d\n", numel(y_train), numel(y_test));

	%% -----------------------------
	% 2) Normalize all predictors for RFF (train stats only)
	% -----------------------------
	X_mu = mean(X_train, 1);
	X_sd = std(X_train, 0, 1);
	X_sd(X_sd < 1e-12) = 1.0;

	X_train_n = (X_train - X_mu) ./ X_sd;
	X_test_n  = (X_test  - X_mu) ./ X_sd;

	%% -----------------------------
	% 3) Fit RFF (resampling) using all predictor dimensions
	% -----------------------------
	d_x = size(X_train_n, 2);
	K = 4000;

	omega_init = 1 * randn(K, d_x);
	omega_sample = omega_init;

	num_resample = 500;
	delta = 0.1;
	Rel_Tol = 1e-3;
	epsilon = 0e-4;
	epsilon_hat = 1e-3;

	J_train = size(X_train_n, 1);
	J_test  = size(X_test_n, 1);

	use_standard_y = false;
	use_log_transform_y = false;

	if use_standard_y
		mu_y = mean(y_train);
		sigma_y = std(y_train);
		y_train_fit = (y_train - mu_y)/sigma_y;
	else
		mu_y = 0; sigma_y = 1;
		y_train_fit = y_train;
	end

	use_early_stopping = true;
	patience = 30;

	lambda = K * sqrt(J_train) / 1e3;

	[omega_used, beta_used, RMSE_test_rff, RMSE_train_rff, ci95_rff] = ...
		rff_resampling_fit( ...
			X_train_n, y_train_fit, ...
			X_test_n,  y_test, ...
			sigma_y, mu_y, J_train, J_test, K, ...
			omega_sample, ...
			use_standard_y, ...
			use_log_transform_y, ...
			use_early_stopping, patience, ...
			num_resample, delta, lambda, Rel_Tol, epsilon, epsilon_hat );

	fprintf("RFF baseline: Train RMSE %.3f, Test RMSE %.3f\n", RMSE_train_rff, RMSE_test_rff);
	fprintf("RFF test 95%% CI: [%.3f, %.3f]\n", ci95_rff(1), ci95_rff(2));

	%% -----------------------------
	% 4) MGAM: PCA on RFF latent features -> GMM -> weighted pyGAM experts
	% -----------------------------
	d_values = [ 5, 6, 7 ];      % PCA dimension
	L_values = [ 12, 13, 14, 15, 16, 17, 18 ];      % clusters

	num_d = numel(d_values);
	num_L = numel(L_values);

	RMSE_train_grid = NaN(num_L, num_d);
	RMSE_test_grid  = NaN(num_L, num_d);
	CI95_lower      = NaN(num_L, num_d);
	CI95_upper      = NaN(num_L, num_d);

	pygam = py.importlib.import_module("pygam");
	np = py.importlib.import_module("numpy");
	s = pygam.s;
	l = pygam.l;

	terms = s(int32(0));
	for j = 1:7
		terms = terms + s(int32(j));
	end

	for d_hat = d_values
		for num_clusters = L_values

			% RFF latent features (TRAIN)
			Phi_train = real(exp(1i * (X_train_n * omega_used')) .* beta_used');  % J_train x K
			h_mean = mean(Phi_train, 1);
			Phi_centered = Phi_train - h_mean;

			[~, ~, V_mat] = svds(Phi_centered, d_hat);
			Z_train = Phi_centered * V_mat;

			std_B = std(Z_train, 0, 1);
			std_B(std_B < 1e-12) = 1.0;
			Z_train_white = Z_train ./ std_B;

			rng(127);
			options = statset('Display','final','MaxIter',1000,'TolFun',1e-6);

			gm = fitgmdist(Z_train_white, num_clusters, ...
				'Options', options, ...
				'CovarianceType','diagonal', ...
				'RegularizationValue', 1e-2, ...
				'Replicates', 20, ...
				'SharedCovariance', false, ...
				'Start','plus');

			idx_hard = cluster(gm, Z_train_white);

			% Posteriors
			Phi_test = real(exp(1i * (X_test_n * omega_used')) .* beta_used');
			Z_test_white = ((Phi_test - h_mean) * V_mat) ./ std_B;

			Gamma_train = posterior(gm, Z_train_white);
			Gamma_test  = posterior(gm, Z_test_white);

			% Fit weighted GAM experts (Gaussian, continuous target)
			X_train_py = np.array(X_train);    % use *un-normalized* X for interpretability of splines
			y_train_py = np.array(y_train);

			gam_models = cell(num_clusters, 1);

			for ell = 1:num_clusters
				w_ell = Gamma_train(:, ell);     % soft weights
				w_ell_py = np.array(w_ell);

				% Guard against degenerate clusters
				if mean(w_ell) < 1e-4
					gam_models{ell} = [];
					continue;
				end

				gam_ell = pygam.LinearGAM(terms);   % Gaussian
				gam_ell = gam_ell.fit(X_train_py, y_train_py, pyargs('weights', w_ell_py));
				gam_models{ell} = gam_ell;
			end

			% Predict mixture
			X_test_py = np.array(X_test);
			Y_pred_test = zeros(J_test, num_clusters);

			for ell = 1:num_clusters
				if isempty(gam_models{ell}), continue; end
				Y_pred_test(:, ell) = double(gam_models{ell}.predict(X_test_py));
			end

			y_hat_test = sum(Gamma_test .* Y_pred_test, 2);
			RMSE_test_MGAM = sqrt(mean((y_hat_test - y_test).^2));
			% fprintf("MGAM: L=%d, d=%d, TEST RMSE=%.3f\n", num_clusters, d_hat, RMSE_test_MGAM);
			
			% ---- Residual bootstrap for test RMSE
			B = 1000;    % number of bootstrap replicates
			resid_test = y_test - y_hat_test;
			
			rmse_boot = zeros(B,1);
			for b = 1:B
				idx_b = randi(J_test, J_test, 1);     % resample indices
				resid_b = resid_test(idx_b);
				y_boot  = y_hat_test + resid_b;       % bootstrap pseudo-response
				rmse_boot(b) = sqrt(mean((y_boot - y_hat_test).^2));
			end
			
			ci95 = quantile(rmse_boot, [0.025, 0.975]);
			
			% ---- Training predictions
			X_train_py = np.array(X_train);
			Y_pred_train = zeros(J_train, num_clusters);
			
			for ell = 1:num_clusters
				if isempty(gam_models{ell}), continue; end
				Y_pred_train(:, ell) = double(gam_models{ell}.predict(X_train_py));
			end
			
			y_hat_train = sum(Gamma_train .* Y_pred_train, 2);
			RMSE_train_MGAM = sqrt(mean((y_hat_train - y_train).^2));
			
			% Store the results on RMSE values
			i_L = find(L_values == num_clusters);
			i_d = find(d_values == d_hat);
			
			RMSE_train_grid(i_L, i_d) = RMSE_train_MGAM;
			RMSE_test_grid(i_L,  i_d) = RMSE_test_MGAM;
			CI95_lower(i_L, i_d)      = ci95(1);
			CI95_upper(i_L, i_d)      = ci95(2);

			fprintf("MGAM L=%d, d=%d | Train RMSE=%.3f | Test RMSE=%.3f | 95%% CI=[%.3f, %.3f]\n", ...
				num_clusters, d_hat, RMSE_train_MGAM, RMSE_test_MGAM, ci95(1), ci95(2));

		end
	end

	% Visualize the results for grid search of (L, d) pairs
	figure();
	imagesc( d_values, L_values, RMSE_test_grid );   % rows = L, cols = d
	colorbar;
	colormap( parula );                        

	xlabel( 'PCA dimension $d$' );
	ylabel( 'Number of GMM clusters $L$' );
	title( 'Test RMSE of MGAM' );

	% Annotate each cell with RMSE
	for i_a = 1 : 1 : num_L
		for i_b = 1 : 1 : num_d
			rmse_val = RMSE_test_grid( i_a, i_b );
			if ~isnan( rmse_val )
				text( d_values( i_b ), L_values( i_a ), sprintf( '%.3f', rmse_val ), ...
					'HorizontalAlignment', 'center', ...
					'Color', 'w', 'FontSize', 10 );
			end
		end
	end		

	%% Export the training and testing dataset
	outdir = fullfile(pwd, 'kin40k_data');
	if ~exist(outdir, 'dir'), mkdir(outdir); end

	writematrix(X_train, fullfile(outdir, 'X_train.csv'));
	writematrix(y_train, fullfile(outdir, 'y_train.csv'));
	writematrix(X_test,  fullfile(outdir, 'X_test.csv'));
	writematrix(y_test,  fullfile(outdir, 'y_test.csv'));

