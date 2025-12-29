
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

	Time3 = [hr_all, weekday_all, mnth_all];   % N × 3   (order is your choice)

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

	%% Resampling/training settings (reuse your conventions)
	num_resample = 300;
	delta = 0.2;
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

	%% ---- Call your RFF resampling trainer (same signature pattern as before)
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
    
   %{
    %% Plot weekday responsibilities of each cluster
    weekday_train = weekday(idx_train);   % values in {0,...,6}
    L = size(Gamma_train, 2);
    days = 0:6;     % Sunday = 0, ..., Saturday = 6
    
    mean_resp_day = zeros(7, L);

    for d = 0:6
        idx_d = (weekday_train == d);
        
        if any(idx_d)
            mean_resp_day(d+1, :) = mean(Gamma_train(idx_d, :), 1);
        else
            mean_resp_day(d+1, :) = NaN;
        end
    end

    figure;
    for ell = 1:L
        subplot(ceil(L/2), 2, ell);
        plot(days, mean_resp_day(:,ell), '-o', 'LineWidth', 1.8);
        
        xlabel('Weekday');
        ylabel(sprintf('\\gamma_%d', ell));
        title(sprintf('Cluster %d', ell));
        
        ylim([0 1]);
        xticks(0:6);
        xticklabels({'Sun','Mon','Tue','Wed','Thu','Fri','Sat'});
        
        grid on;
    end

    %% Plot month responsibilities of each cluster
    mnth_train = mnth(idx_train);   % values in {1,...,12}
    L = size(Gamma_train, 2);
    months = 1:12;
    
    mean_resp_month = zeros(12, L);

    for m = 1:12
        idx_m = (mnth_train == m);
        
        if any(idx_m)
            mean_resp_month(m, :) = mean(Gamma_train(idx_m, :), 1);
        else
            mean_resp_month(m, :) = NaN;
        end
    end

	figure;
    for ell = 1:L
        subplot(ceil(L/2), 2, ell);
        plot(months, mean_resp_month(:,ell), '-o', 'LineWidth', 1.8);
        
        xlabel('Month');
        ylabel(sprintf('\\gamma_%d', ell));
        title(sprintf('Cluster %d', ell));
        
        ylim([0 1]);
        xticks(1:12);
        xticklabels({'Jan','Feb','Mar','Apr','May','Jun', ...
                      'Jul','Aug','Sep','Oct','Nov','Dec'});
        
        grid on;
    end
    
    %% Plot working day responsibilities
    workingday_train = workingday(idx_train);   % 1 = working day, 0 = non-working
    L = size(Gamma_train, 2);

    mean_resp_work    = zeros(1, L);
    mean_resp_nonwork = zeros(1, L);
    
    idx_work    = (workingday_train == 1);
    idx_nonwork = (workingday_train == 0);
    
    mean_resp_work    = mean(Gamma_train(idx_work, :), 1);
    mean_resp_nonwork = mean(Gamma_train(idx_nonwork, :), 1);
    
    clusters = 1:L;

    figure;
    
    % --- Working day ---
    subplot(1,2,1);
    bar(clusters, mean_resp_work);
    ylim([0 1]);
    xlabel('Cluster index');
    ylabel('Average posterior responsibility');
    title('Working days');
    grid on;
    
    % --- Non-working day ---
    subplot(1,2,2);
    bar(clusters, mean_resp_nonwork);
    ylim([0 1]);
    xlabel('Cluster index');
    ylabel('Average posterior responsibility');
    title('Non-working days');
    grid on;

    %% Plot of responsibilities for two categories, working day or non-working day
    hr_train = hr(idx_train);                 % 0,...,23
    workingday_train = workingday(idx_train); % 1 = working, 0 = non-working
    
    L = size(Gamma_train, 2);
    hours = 0:23;
    
    mean_resp_hour_work    = zeros(24, L);
    mean_resp_hour_nonwork = zeros(24, L);

    for h = 0:23
        idx_h = (hr_train == h);
        
        idx_hw = idx_h & (workingday_train == 1);
        idx_hn = idx_h & (workingday_train == 0);
        
        if any(idx_hw)
            mean_resp_hour_work(h+1, :) = mean(Gamma_train(idx_hw, :), 1);
        else
            mean_resp_hour_work(h+1, :) = NaN;
        end
        
        if any(idx_hn)
            mean_resp_hour_nonwork(h+1, :) = mean(Gamma_train(idx_hn, :), 1);
        else
            mean_resp_hour_nonwork(h+1, :) = NaN;
        end
    end
    
    figure;
    for ell = 1:L
        subplot(ceil(L/2), 2, ell);
        
        plot(hours, mean_resp_hour_work(:,ell), '-o', ...
             'LineWidth', 1.8); hold on;
        plot(hours, mean_resp_hour_nonwork(:,ell), '--s', ...
             'LineWidth', 1.8);
        
        xlabel('Hour of day');
        ylabel(sprintf('\\gamma_%d', ell));
        title(sprintf('Cluster %d', ell));
        ylim([0 1]);
        
        legend({'Working day','Non-working day'}, 'Location','best');
        grid on;
    end

    %% Plot the responsibilities for each hour in four seasons
    mnth_train = mnth(idx_train);   % values 1,...,12

    season_id = zeros(size(mnth_train));
    
    % Winter: Dec–Feb
    season_id(ismember(mnth_train, [12,1,2])) = 1;
    
    % Spring: Mar–May
    season_id(ismember(mnth_train, [3,4,5])) = 2;
    
    % Summer: Jun–Aug
    season_id(ismember(mnth_train, [6,7,8])) = 3;
    
    % Autumn: Sep–Nov
    season_id(ismember(mnth_train, [9,10,11])) = 4;

    hr_train = hr(idx_train);       % 0,...,23
    L = size(Gamma_train, 2);
    hours = 0:23;
    
    mean_resp_hour_season = zeros(24, L, 4);

    for h = 0:23
        idx_h = (hr_train == h);
        
        for s = 1:4
            idx_hs = idx_h & (season_id == s);
            
            if any(idx_hs)
                mean_resp_hour_season(h+1, :, s) = mean(Gamma_train(idx_hs, :), 1);
            else
                mean_resp_hour_season(h+1, :, s) = NaN;
            end
        end
    end

    season_labels = {'Winter','Spring','Summer','Autumn'};
    colors = lines(4);
    
    figure;
    for ell = 1:L
        subplot(ceil(L/2), 2, ell);
        hold on;
        
        for s = 1:4
            plot(hours, mean_resp_hour_season(:,ell,s), ...
                 '-o', 'LineWidth', 1.6, 'Color', colors(s,:));
        end
        
        xlabel('Hour of day');
        ylabel(sprintf('\\gamma_%d', ell));
        title(sprintf('Cluster %d', ell));
        ylim([0 1]);
        
        legend(season_labels, 'Location', 'best');
        grid on;
    end
    
    %% Plot responsibilities under different weather situations
    weather_train = weathersit(idx_train);
    hr_train = hr(idx_train);
    L = size(Gamma_train, 2);
    hours = 0:23;
    num_weather = 4;
    
    mean_resp_hour_weather = zeros(24, L, num_weather);

    for h = 0:23
        idx_h = (hr_train == h);
        
        for w = 1:num_weather
            idx_hw = idx_h & (weather_train == w);
            
            if any(idx_hw)
                mean_resp_hour_weather(h+1, :, w) = mean(Gamma_train(idx_hw, :), 1);
            else
                mean_resp_hour_weather(h+1, :, w) = NaN;
            end
        end
    end

    weather_labels = {'Clear','Mist/Cloudy','Light rain/snow','Heavy rain/snow'};
    colors = lines(4);
    
    figure;
    for ell = 1:L
        subplot(ceil(L/2), 2, ell);
        hold on;
        
        for w = 1:4
            plot(hours, mean_resp_hour_weather(:,ell,w), ...
                'grid-o', 'LineWidth', 1.6, 'Color', colors(w,:));
        end
        
        xlabel('Hour of day');
        ylabel(sprintf('\\gamma_%d', ell));
        title(sprintf('Cluster %d', ell));
        ylim([0 1]);
        legend(weather_labels, 'Location', 'best');
        grid on;
    end
	
	%% Plot the partial dependence on the four continuous variables
	cont_names = {'temp','atemp','hum','windspeed'};
	cont_cols  = [8, 9, 10, 11];   % confirm these match your construction
	
	csv_files = { ...
		'mlm_pdp_temp.csv', ...
		'mlm_pdp_atemp.csv', ...
		'mlm_pdp_hum.csv', ...
		'mlm_pdp_windspeed.csv' };

	mlm_x   = cell(4,1);
	mlm_pdp = cell(4,1);

	for j = 1:4
		Tref = readtable(csv_files{j});
		mlm_x{j}   = Tref{:,1};
		mlm_pdp{j} = Tref{:,2};
	end
    
	ngrid = 10;
	grid_vals = cell(4,1);

	for j = 1:4
		
		% xmin = min(mlm_x{j});
		% xmax = max(mlm_x{j});
		xj = X_train(:, cont_cols(j));
		xmin = max(min(mlm_x{j}), prctile(xj, 1));
		xmax = min(max(mlm_x{j}), prctile(xj, 99));
		grid_vals{j} = linspace(xmin, xmax, ngrid);
		% xj = X_train(:, cont_cols(j));
		% grid_vals{j} = linspace(min(xj), max(xj), ngrid);
	end

	np = py.importlib.import_module("numpy");
	PDP_mgam = cell(4,1);
	for j = 1:4

		grid_pts = grid_vals{j};
		col  = cont_cols(j);

		pdp_j = zeros(numel(grid_pts),1);

		for k = 1:numel(grid_pts)

			% Intervene on one variable
			Xk = X_train;
			Xk(:, col) = grid_pts(k);

			Xk_py = np.array(Xk);

			% Mixture prediction
			y_mix = zeros(size(X_train,1),1);

			for ell = 1:size(Gamma_train,2)
				if isempty(gam_models{ell})
					continue;
				end
				y_ell = double(gam_models{ell}.predict(Xk_py));
				y_mix = y_mix + Gamma_train(:,ell) .* y_ell';
			end

			% Average over samples
			pdp_j(k) = mean(y_mix);
		end

		PDP_mgam{j} = pdp_j;
	end
	
	figure('Color','w','Position',[100 100 1200 300]);
	for j = 1:4
		subplot(1,4,j); hold on;

		% Mixture-of-GAMs PDP
		plot(grid_vals{j}, PDP_mgam{j}, ...
			'LineWidth',2, 'Color',[0 0.45 0.74]);

		% MLM reference PDP
		plot(mlm_x{j}, mlm_pdp{j}, '--', ...
			'LineWidth',2, 'Color',[0.85 0.33 0.1]);

		xlabel(cont_names{j});
		ylabel('Partial dependence');
		title(cont_names{j});
		grid on;

		if j == 1
			legend({'Mixture of GAMs','Mixture of Linear Models'}, ...
				'Location','best');
		end
	end

	% Mixture weights sanity check
	fprintf('Max |sum Gamma − 1| = %.2e\n', ...
		max(abs(sum(Gamma_train,2) - 1)));

	% PDP scale sanity check
	fprintf('Mean prediction vs PDP(temp) mean: %.2f vs %.2f\n', ...
		mean(PDP_mgam{1}), mean(PDP_mgam{1}));
		
   %}

	
	
	
			
			
			