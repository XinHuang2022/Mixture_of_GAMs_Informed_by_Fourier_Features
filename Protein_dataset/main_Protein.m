	clear

	addpath(genpath('utilities'));

	%% -----------------------------
	% 1) Load raw data (UCI format)
	% -----------------------------
	data_file = fullfile(pwd, 'data', 'data.csv');
	test_mask_file = fullfile(pwd, 'data', 'test_mask.csv');

	% Load
	data_all = readmatrix(data_file);
	% Load mask as matrix
	mask_matrix = readmatrix(test_mask_file);   % 16599 x 18
	% Use column 1 as the test set
	mask_test = logical(mask_matrix(:,1));  % 16599 x 1 logical

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
	K = 12000;

	omega_init = 1 * randn(K, d_x);
	omega_sample = omega_init;

	num_resample = 500;
	delta = 0.3;
	Rel_Tol = 1e-2;
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
    
    save('trained_RFF_params.mat', 'beta_used', 'omega_used')
	%% -----------------------------
	% 4) MGAM: PCA on RFF latent features -> GMM -> weighted pyGAM experts
	% -----------------------------
	% d_values = [ 3, 5, 7 ];      % PCA dimension
	% L_values = [ 6, 8, 10, 12 ];      % clusters
    d_values = [ 6, 8 ];
	L_values = [ 22, 24 ];

	num_d = numel(d_values);
	num_L = numel(L_values);

	RMSE_train_grid = NaN(num_L, num_d);
	RMSE_test_grid  = NaN(num_L, num_d);
	CI95_lower      = NaN(num_L, num_d);
	CI95_upper      = NaN(num_L, num_d);
    
    load('trained_RFF_params_K=12000_delta=0_3.mat', 'beta_used', 'omega_used');
    % Precompute train/test RFF feature matrices once
	Phi_train_full = real(exp(1i * (X_train_n * omega_used')) .* beta_used');   % J_train x K
	Phi_test_full  = real(exp(1i * (X_test_n  * omega_used')) .* beta_used');   % J_test  x K

	for d_hat = d_values
		for num_clusters = L_values

			% PCA on RFF latent features (TRAIN)
			h_mean = mean(Phi_train_full, 1);
			Phi_centered = Phi_train_full - h_mean;

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

			% Hard labels for training subsets
			idx_hard = cluster(gm, Z_train_white);

			% Soft posteriors for mixture prediction
			Z_test_white = ((Phi_test_full - h_mean) * V_mat) ./ std_B;

			Gamma_train = posterior(gm, Z_train_white);
			Gamma_test  = posterior(gm, Z_test_white);

			% Optional diagnostic
			cluster_sizes = histcounts(idx_hard, 1:(num_clusters+1));
			disp('Hard cluster sizes:');
			disp(cluster_sizes);

			% ---- Hard-cluster GAM experts using custom backfitting
			gam_models_list = cell(num_clusters, 1);
			alpha_vals = zeros(1, num_clusters);

			lambda_gam   = 1e0;
			max_iter_gam = 300;
			knots_per_dim = 10;
			degree_gam   = 3;
			rel_tol_gam  = 1e-3;

			for ell = 1:num_clusters
				idx_ell = (idx_hard == ell);

				% Guard against tiny clusters
				if sum(idx_ell) < 20
					gam_models_list{ell} = [];
					alpha_vals(ell) = NaN;
					continue;
				end

				X_ell = X_train_n(idx_ell, :);
				y_ell = y_train(idx_ell);

				[alpha_ell, f_list_ell, delta_ell] = fit_gam_backfitting( ...
					X_ell, y_ell, knots_per_dim, lambda_gam, ...
					max_iter_gam, degree_gam, rel_tol_gam );

				gam_models_list{ell} = f_list_ell;
				alpha_vals(ell) = alpha_ell;
			end

			% ---- Test predictions from all local GAM experts
			Y_pred_test = zeros(J_test, num_clusters);

			for ell = 1:num_clusters
				if isempty(gam_models_list{ell}), continue; end

				total_pred_test = zeros(J_test, 1);
				f_list_ell = gam_models_list{ell};

				for feature_index = 1:d_x
					x_feature_vals = X_test_n(:, feature_index);
					Phi_feature_index = build_bspline_basis( ...
						x_feature_vals, ...
						f_list_ell{feature_index}.full_knots, ...
						degree_gam );
					total_pred_test = total_pred_test + Phi_feature_index * f_list_ell{feature_index}.theta;
				end

				Y_pred_test(:, ell) = alpha_vals(ell) + total_pred_test;
			end

			% Soft mixture prediction on test data
			y_hat_test = sum(Gamma_test .* Y_pred_test, 2);
			RMSE_test_MGAM = sqrt(mean((y_hat_test - y_test).^2));

			% ---- Residual bootstrap for test RMSE
			B = 1000;
			resid_test = y_test - y_hat_test;

			rmse_boot = zeros(B,1);
			for b = 1:B
				idx_b = randi(J_test, J_test, 1);
				resid_b = resid_test(idx_b);
				y_boot  = y_hat_test + resid_b;
				rmse_boot(b) = sqrt(mean((y_boot - y_hat_test).^2));
			end

			ci95 = quantile(rmse_boot, [0.025, 0.975]);

			% ---- Training predictions from all local GAM experts
			Y_pred_train = zeros(J_train, num_clusters);

			for ell = 1:num_clusters
				if isempty(gam_models_list{ell}), continue; end

				total_pred_train = zeros(J_train, 1);
				f_list_ell = gam_models_list{ell};

				for feature_index = 1:d_x
					x_feature_vals = X_train_n(:, feature_index);
					Phi_feature_index = build_bspline_basis( ...
						x_feature_vals, ...
						f_list_ell{feature_index}.full_knots, ...
						degree_gam );
					total_pred_train = total_pred_train + Phi_feature_index * f_list_ell{feature_index}.theta;
				end

				Y_pred_train(:, ell) = alpha_vals(ell) + total_pred_train;
			end

			% Soft mixture prediction on training data
			y_hat_train = sum(Gamma_train .* Y_pred_train, 2);
			RMSE_train_MGAM = sqrt(mean((y_hat_train - y_train).^2));

			% ---- Store results
			i_L = find(L_values == num_clusters);
			i_d = find(d_values == d_hat);

			RMSE_train_grid(i_L, i_d) = RMSE_train_MGAM;
			RMSE_test_grid(i_L,  i_d) = RMSE_test_MGAM;
			CI95_lower(i_L, i_d)      = ci95(1);
			CI95_upper(i_L, i_d)      = ci95(2);

			fprintf("MGAM-hard L=%d, d=%d | Train RMSE=%.3f | Test RMSE=%.3f | 95%% CI=[%.3f, %.3f]\n", ...
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
	outdir = fullfile(pwd, 'Protein_data');
	if ~exist(outdir, 'dir'), mkdir(outdir); end

	writematrix(X_train, fullfile(outdir, 'X_train.csv'));
	writematrix(y_train, fullfile(outdir, 'y_train.csv'));
	writematrix(X_test,  fullfile(outdir, 'X_test.csv'));
	writematrix(y_test,  fullfile(outdir, 'y_test.csv'));

