
    clear

    %% Configure the virtual environment with correct Python version (3.10)
    pyenv( "Version", "G:\Research\Projects\Post_Graduate\Explainable_Machine_Learning\Random_Feature_Models\codes_backup\pygam_virt_env\Scripts\python.exe", "ExecutionMode", "OutOfProcess" )


	%% Load raw data
	data_file = fullfile( pwd, 'data', 'hour.csv' );
	T = readtable( data_file );
	% Add path for helper functions
	addpath( genpath( 'utilities' ) );

	% Target
	y_all = T.cnt;

	% Time variables used for RFF clustering (3D)
	mnth_all    = T.mnth;      % usually 1..12
	hr_all      = T.hr;        % 0..23
	weekday_all = T.weekday;   % 0..6
    day_all = day(datetime(T.dteday));   % values 1..31

	Time3 = [hr_all, weekday_all, mnth_all];   % N × 3  

	%% Split train/test
	rng(127);
	N = size(Time3, 1);
	idx = randperm(N);

	test_ratio = 0.2;
	n_test  = round(test_ratio * N);
	idx_test  = idx(1:n_test);
	idx_train = idx(n_test+1:end);

	Time3_train = Time3(idx_train, :);
	Time3_test  = Time3(idx_test,  :);

	y_train = y_all(idx_train);
	y_test  = y_all(idx_test);
	
	%% Extract raw variables
	yr = T.yr;
	mnth = T.mnth;
	hr = T.hr;
	weekday = T.weekday;
    monthday = day(datetime(T.dteday));   % 1..31

	holiday = T.holiday;
	workingday = T.workingday;

	temp = T.temp;
	atemp = T.atemp;
	hum = T.hum;
	windspeed = T.windspeed;

	season = T.season;
	weathersit = T.weathersit;
	
	% Binary variables (keep as-is)
	X_bin = [yr, holiday, workingday];

	% Cyclic variables
	X_cyc = [mnth, hr, weekday, monthday];

	% Continuous smooth variables
	X_cont = [temp, atemp, hum, windspeed];

	% Dummy-encoded categorical variables
	season_d  = dummyvar(season);     	season_d  = season_d(:,2:end);
	weather_d = dummyvar(weathersit); 	weather_d = weather_d(:,2:end);

	% Final design matrix
	X = [X_bin, X_cyc, X_cont, season_d, weather_d];
	X_train = X(idx_train,:);
	X_test = X(idx_test,:);

	fprintf("Train size: %d, Test size: %d\n", numel(y_train), numel(y_test));

	%% Normalize the 3 time variables using training statistics only
    
	Time_mu = mean(Time3_train, 1);
	Time_sd = std(Time3_train, 0, 1);

	% (safety) avoid division by 0
	Time_sd(Time_sd < 1e-12) = 1.0;

	Time3_train_n = (Time3_train - Time_mu) ./ Time_sd;
	Time3_test_n  = (Time3_test  - Time_mu) ./ Time_sd;
    
	%% RFF hyperparameters (analogous to spatial-RFF block)
	d_time = 3;
	K = 2000;                    % number of random features / frequencies
	use_init_omega_zero = false;

	if use_init_omega_zero
		omega_init = zeros(K, d_time);
	else
		omega_init = randn(K, d_time);
	end
	omega_sample_time = omega_init;

	%% Resampling/training settings
	num_resample = 500;
	delta = 0.05;
	Rel_Tol = 1e-3;
	epsilon = 0;
	epsilon_hat = 1e-3;

	J_train = size(Time3_train_n,1);
	J_test  = size(Time3_test_n,1);

	use_standard_y = false;
	use_log_transform_y = false;
	if use_standard_y
		mu_y = mean(y_train);
		sigma_y = std(y_train);
		y_train_fit = (y_train - mu_y) / sigma_y;
	else
		mu_y = 0;
		sigma_y = 1;
		y_train_fit = y_train;
	end

	use_early_stopping = true;
	patience = 30;

	lambda = K * sqrt(J_train) / 100;

	%% Call the RFF resampling trainer
	[omega_used_time, beta_used_time, RMSE_test_rff, RMSE_train_rff, ci95_rff] = ...
		rff_resampling_fit( ...
			Time3_train_n, y_train_fit, ...
			Time3_test_n,  y_test, ...
			sigma_y, mu_y, J_train, J_test, K, ...
			omega_sample_time, ...
			use_standard_y, ...                
			use_log_transform_y, ...               
			use_early_stopping, patience, ...
			num_resample, delta, lambda, Rel_Tol, epsilon, epsilon_hat );

	fprintf("Temporal-RFF baseline: Train RMSE %.3f, Test RMSE %.3f\n", RMSE_train_rff, RMSE_test_rff);
	fprintf("Temporal-RFF test 95%% CI: [%.3f, %.3f]\n", ci95_rff(1), ci95_rff(2));
	
	%% Grid search for Mixture-of-GAMs hyperparameters d and L
    % d_values = [2, 3, 4, 5];
	% L_values = [4, 5, 6, 7, 8, 9];
    d_values = 3;
    L_values = 8;

	num_d = length(d_values);
	num_L = length(L_values);

	RMSE_grid_time = zeros(num_L, num_d);
	RMSE_grid_train_time = zeros(num_L, num_d);

	best_RMSE = Inf;
	best_gm = [];
	best_V = [];
	best_h_mean = [];
	best_std_B = [];
	
	% Verify pyGAM is visible
	pygam = py.importlib.import_module("pygam");

    % Also import Python module numpy
    np = py.importlib.import_module("numpy");
	
	% Import pyGAM components
    s = pygam.s;    % s(j) = spline term on column j
    l = pygam.l;    % l(j) = linear term on column j
    
    % Cyclic spline terms
	terms = ...
    s(int32(4-1), pyargs("basis", "cp", "n_splines", int32(12))) + ... % mnth
    s(int32(5-1), pyargs("basis", "cp", "n_splines", int32(24))) + ... % hr
    s(int32(6-1), pyargs("basis", "cp", "n_splines", int32(7))); % weekday
   
	% Smooth (non-cyclic) splines
	terms = terms + ...
    s(int32(8-1)) + ...
    s(int32(9-1)) + ...
    s(int32(10-1)) + ...
    s(int32(11-1));

    % Linear terms (binary + dummy variables)
    linear_cols = [1,2,3,12:17];
    for j = linear_cols
        terms = terms + l(int32(j-1));
    end
    
    % Start grid search for Mixture-of-GAMs model's setting
	for i_a = 1:num_L
		for i_b = 1:num_d
			
			d_hat = d_values(i_b);
			num_clusters = L_values(i_a);
			% Temporal RFF latent features (TRAIN)
			Phi_train = real(exp(1i * (Time3_train_n * omega_used_time')) .* beta_used_time');   % J_train × K
			h_mean = mean(Phi_train, 1);
			Phi_centered = Phi_train - h_mean;

			[~, ~, V_mat] = svds(Phi_centered, d_hat);
			Z_train = Phi_centered * V_mat;   % J_train × d_hat
			
			std_B = std(Z_train, 0, 1);
			std_B(std_B < 1e-12) = 1.0;
			Z_train_white = Z_train ./ std_B;

            rng(127 + 20*i_a + 10*i_b, 'twister');
			
			options = statset('Display','final','MaxIter',1000,'TolFun',1e-6);

			gm = fitgmdist( ...
				Z_train_white, num_clusters, ...
				'Options', options, ...
				'CovarianceType', 'diagonal', ...
				'RegularizationValue', 1e-2, ...
				'Replicates', 30, ...
				'SharedCovariance', false, ...
				'Start', 'plus' ...
			);

			idx = cluster(gm, Z_train_white);
            cluster_sizes = accumarray(idx, 1, [num_clusters 1]);
            min_cluster_size = 200;   % or 1–2% of training data

            if any(cluster_sizes < min_cluster_size)
                fprintf('GMM model fitting rejected due to too small cluster size.\n');
            end

			
			Phi_test = real(exp(1i * (Time3_test_n * omega_used_time')) .* beta_used_time');   % J_test × K
			Phi_test_centered = Phi_test - h_mean;
			Z_test = Phi_test_centered * V_mat;
			Z_test_white = Z_test ./ std_B;
			
			Gamma_test = posterior(gm, Z_test_white);   % J_test × num_clusters
            Gamma_train = posterior(gm, Z_train_white); % J_train × num_clusters

			gam_models = cell(num_clusters, 1);
			train_rmse_per_cluster = zeros(num_clusters, 1);

			for ell = 1:num_clusters
				% indices for cluster ell (hard assignment)
				idx_ell = find(idx == ell);

				% Guard against tiny clusters
				if numel(idx_ell) < 50
					warning("Cluster %d too small (%d points). Skipping.", ell, numel(idx_ell));
					gam_models{ell} = [];
					continue;
				end

				% training data for cluster ell
				% X_ell = X_train(idx_ell, :);   % original covariates
				% y_ell = y_train(idx_ell);  % original cnt

                weights_ell = Gamma_train(:, ell);

				% ---- convert to numpy
				% X_ell_py = np.array(X_ell);
				% y_ell_py = np.array(y_ell);
                X_train_py = np.array(X_train);
                y_train_py = np.array(y_train);
                w_ell_py = np.array(weights_ell);
                % weights_ell = weights_ell / mean(weights_ell(weights_ell > 0));

				% ---- fit pyGAM expert
                
				gam_ell = pygam.PoissonGAM(terms);
                gam_ell = gam_ell.fit(X_train_py, y_train_py, pyargs('weights', w_ell_py));

				gam_models{ell} = gam_ell;
              	
			end
			
			% Evaluate the test RMSE of Mixture-of-GAMs model
			Y_pred_test = zeros(J_test, num_clusters);
			X_test_py = np.array(X_test);
			for ell = 1:num_clusters
				if isempty(gam_models{ell})
					continue;
				end

				% Predict mean count
				Y_pred_test(:, ell) = double(gam_models{ell}.predict(X_test_py));
			end
			% Gamma_test: J_test × num_clusters
			y_hat_test = sum(Gamma_test .* Y_pred_test, 2);
			RMSE_test_MGAM = sqrt(mean((y_hat_test - y_test).^2));
			fprintf("L = %d, d = %d, Mixture-of-GAMs TEST RMSE = %.3f\n", num_clusters, d_hat, RMSE_test_MGAM);

            % Number of bootstrap replicates
            B = 1000;
            % Residuals on test set
            resid = y_test - y_hat_test;
            rmse_boot = zeros(B,1);
            for b = 1:B
                % Resample residuals with replacement
                idx_b = randi(J_test, J_test, 1);
                resid_b = resid(idx_b);
            
                % Pseudo test observations
                y_boot = y_hat_test + resid_b;
            
                % Bootstrap RMSE
                rmse_boot(b) = sqrt(mean((y_boot - y_hat_test).^2));
            end
            % 95% confidence interval
            ci95 = quantile(rmse_boot, [0.025, 0.975]);
            fprintf('Test RMSE = %.3f\n', RMSE_test_MGAM);
            fprintf('95%% CI = [%.3f, %.3f]\n', ci95(1), ci95(2));
			
			% Evaluate the training RMSE of Mixture-of-GAMs model
			Y_pred_train = zeros(J_train, num_clusters);
			X_train_py = np.array(X_train);

			for ell = 1:num_clusters
				if isempty(gam_models{ell})
					continue;
				end
				Y_pred_train(:, ell) = double(gam_models{ell}.predict(X_train_py));
			end
			
			y_hat_train = sum(Gamma_train .* Y_pred_train, 2);

			RMSE_train_MGAM = sqrt(mean((y_hat_train - y_train).^2));

			fprintf("Mixture-of-GAMs TRAIN RMSE = %.3f\n", RMSE_train_MGAM);

			RMSE_grid_time(i_a, i_b)  = RMSE_test_MGAM;
			RMSE_grid_train_time(i_a, i_b) = RMSE_train_MGAM;

		end	
	end		
			
	figure();
    imagesc( d_values, L_values, RMSE_grid_time );   % rows = L, cols = d
    colorbar;
    colormap( parula );                        
    
    xlabel( 'PCA dimension $d$' );
    ylabel( 'Number of GMM clusters $L$' );
    title( 'Test RMSE of MGAM' );

    % Annotate each cell with RMSE
    for i_a = 1 : 1 : num_L
        for i_b = 1 : 1 : num_d
            rmse_val = RMSE_grid_time( i_a, i_b );
            if ~isnan( rmse_val )
                text( d_values( i_b ), L_values( i_a ), sprintf( '%.3f', rmse_val ), ...
                    'HorizontalAlignment', 'center', ...
                    'Color', 'w', 'FontSize', 10 );
            end
        end
    end		
	
    %% Exploratory analysis on the GMM-clustered data
    % Plot hour responsibilities of each cluster
	hr_train = hr(idx_train);   % values in {0,...,23}
	L = size(Gamma_train, 2);
    hours = 0:23;
    
    mean_resp_hour = zeros(24, L);
    for h = 0:23
        idx_h = (hr_train == h);
        
        if any(idx_h)
            mean_resp_hour(h+1, :) = mean(Gamma_train(idx_h, :), 1);
        else
            mean_resp_hour(h+1, :) = NaN;
        end
    end

    figure;
    for ell = 1:L
        subplot(ceil(L/2), 2, ell);
        plot(hours, mean_resp_hour(:,ell), '-o', 'LineWidth', 1.8);
        xlabel('Hour');
        ylabel(sprintf('\\gamma_%d', ell));
        title(sprintf('Cluster %d', ell));
        ylim([0 1]);
        grid on;
    end
    
   
	
	
			
			
			