	
    clear 

	%% Preload data
	data_file = fullfile( pwd, 'data', 'airfoil_self_noise.dat' );
	Airfoil_array = readmatrix( data_file, 'FileType', 'text' );
	Airfoil_array = Airfoil_array( ~any( isnan( Airfoil_array ), 2 ), : );
    Airfoil_array( :, 1 ) = log( Airfoil_array( :, 1 ) );    % Logarithmic-transform on the frequency variable
	J = size( Airfoil_array, 1 );
	% Add path for helper functions
	addpath( genpath( 'utilities' ) );
	

	rng( 127 );
	rand_ind = randperm( J );
	test_ratio = 0.2;
	J_train = round( ( 1 - test_ratio ) * J );
	J_test = J - J_train;
	train_idx = rand_ind( 1 : J_train );
	test_idx  = rand_ind( J_train + 1 : J );

	X_train = Airfoil_array( train_idx, 1 : 5 );  
	y_train_ori = Airfoil_array( train_idx, 6 );
	X_test  = Airfoil_array( test_idx, 1 : 5 );
	y_test_ori = Airfoil_array( test_idx, 6 );

	% === Standardize ===
	mu_x = mean( X_train, 1 );
	sigma_x = std( X_train, 0, 1 );
	X_train_norm = ( X_train - mu_x ) ./ sigma_x;
	X_test_norm  = ( X_test - mu_x ) ./ sigma_x;
	mu_y = mean( y_train_ori );
	sigma_y = std( y_train_ori );
	y_train_norm = ( y_train_ori - mu_y ) / sigma_y;
	
	x_data_train = X_train_norm;
	y_data_train = y_train_norm;
	x_data_test = X_test_norm;
	y_data_test_ori = y_test_ori;
	
	
	d = 5;
	K = 2000;
	
	% Determine the initial distribution of frequency parameters
	use_init_omega_zero = false;
	use_standard_data = true;
	use_log_transform_y = false;
	use_early_stopping = true;
	patience = 30;
	if( use_init_omega_zero )
		omega_init = zeros( K, d );
	else
		omega_init = 3 * randn( K, d );
	end
	omega_sample = omega_init;
	num_resample = 300;
	delta = 0.1;
	lambda = K * sqrt( J_train ) / 1000;
	Rel_Tol = 1 * 10^( -3 ); 
	epsilon = 0;
	C_mat_init = eye( d );
	C_mat_sample = C_mat_init;
	epsilon_hat = 1e-3;
	
	[ omega_used, beta_used, RMSE_test, RMSE_train, ci95 ] = ...
	rff_resampling_fit( X_train_norm, y_train_norm, X_test_norm, y_test_ori, ...
	sigma_y, mu_y, J_train, J_test, K, ...
	omega_sample, use_standard_data, use_log_transform_y, ...
	use_early_stopping, patience, ...
	num_resample, delta, lambda, Rel_Tol, epsilon, epsilon_hat );
	
	d_hat = 2;
	num_clusters = 12;

	% Build RFF features on train/test
	Mix_mat_A = real( exp( 1i * ( X_train_norm * omega_used' ) ) .* beta_used' );
	h_mean = mean( Mix_mat_A, 1 );
	[ U, S, V ] = svds( Mix_mat_A - h_mean, d_hat );
	Reduce_dim_train = ( Mix_mat_A - h_mean ) * V;
	std_B = std( Reduce_dim_train, 0, 1 );
	Reduce_dim_train_white = Reduce_dim_train ./ std_B;

	% Fit GMM
	gm = fitgmdist( Reduce_dim_train_white,num_clusters, ...
		'CovarianceType', 'diagonal', 'RegularizationValue', 1e-3, ...
		'Replicates', 20, 'Start', 'plus', 'Options', statset( 'MaxIter', 2000 ) );

	idx = cluster( gm, Reduce_dim_train_white );

	% Fit GAM per cluster
	gam_models_list = cell( num_clusters, 1 );
	alpha_vals = zeros( 1, num_clusters );
	for ell = 1 : 1 : num_clusters
		idx_ell = find( idx == ell );
		X_ell = X_train_norm( idx_ell, : );
		y_ell = y_train_norm( idx_ell );
		[ alpha_ell, f_list_ell, ~ ] = fit_gam_backfitting( X_ell, y_ell, 50, 0.1, 300, 3, 1e-3 );
		gam_models_list{ell} = f_list_ell;
		alpha_vals( 1, ell ) = alpha_ell;
	end

	% Predict on test set
	Mix_mat_test = real( exp( 1i * ( X_test_norm * omega_used' ) ).* beta_used' );
	Mix_centered_test = Mix_mat_test - h_mean;
	Reduce_dim_test = Mix_centered_test * V;
	Reduce_dim_test_white = Reduce_dim_test ./ std_B;
	Gamma_mat = posterior( gm, Reduce_dim_test_white );

	% compute predictions
	dfeat = size( X_test_norm, 2 );
	Y_preds_std_test = zeros( J_test, num_clusters );
	for ell = 1 : 1 : num_clusters
		total_pred_test = zeros( J_test, 1 );
		f_list_ell = gam_models_list{ell};
		for feature_index = 1 : 1 : dfeat
			x_feature_vals = X_test_norm( :, feature_index );
			Phi_feature_index = build_bspline_basis( x_feature_vals, f_list_ell{feature_index}.full_knots, 3 );
			total_pred_test = total_pred_test + Phi_feature_index * f_list_ell{feature_index}.theta;
		end
		Y_preds_std_test( :, ell ) = alpha_vals( 1, ell ) + total_pred_test;
	end
	y_hat_std_test = sum( Gamma_mat .* Y_preds_std_test, 2 );
	y_hat_test = y_hat_std_test * sigma_y + mu_y;
	RMSE_test_MGAM = sqrt( mean( ( y_hat_test - y_test_ori ).^2 ) );
    fprintf( 'RMSE of Mixture of GAMs: %5.4e\n', RMSE_test_MGAM );

    %% Partial Dependence Plots for RFF regression model with resampled Fourier features
    feature_names = { ...
        'Acoustic Frequency (Hz)', ...              % Feature 1 (log-transformed)
        'Angle of Attack', ...
        'Chord Length', ...
        'Free-stream Velocity', ...
        'Suction-side Displacement Thickness' };
    
    num_plot = 30;      % grid resolution for PDP
    N = size( X_train_norm, 1 );
    figure_counter = 1;
    
    for feat = 1:5
    
        % PDP evaluation grid on normalized scale
        x_min = min( X_train_norm( :, feat ) );
        x_max = max( X_train_norm( :, feat ) );
        x_feat_grid = linspace( x_min, x_max, num_plot );
    
        y_partial = zeros( 1, num_plot );
    
        for np = 1:num_plot
    
            % clamp feature feat to grid value
            X_mod = X_train_norm;
            X_mod( :, feat ) = x_feat_grid( np );
    
            % RFF feature construction
            % RFF feature for each sample: real(exp(i x^T w)) * beta
            Z = real( exp( 1i * ( X_mod * omega_used' ) ) * beta_used );   % N × 1
    
            % rescale y back to original scale
            y_mod = Z * sigma_y + mu_y;
    
            % PDP value = mean prediction
            y_partial( 1, np ) = mean( y_mod );
        end
    
        % Plot PDP
        figure(); figure_counter = figure_counter + 1;
        hold on
        % Convert x-axis back to ORIGINAL scale
        x_feat_orig = x_feat_grid * sigma_x( feat ) + mu_x( feat );
    
        if feat == 1
            % Feature 1 = frequency (log-transformed)
            freq_Hz = exp( x_feat_orig );    % convert from log-scale
            semilogx( freq_Hz, y_partial, 'LineWidth', 2 );
            xlabel( 'Frequency (Hz) [original scale]' );
        else
            plot( x_feat_orig, y_partial, 'LineWidth', 2 );
            xlabel( feature_names{feat} );
        end
    
        ylabel( 'Predicted SPL (dB)' );
        title( ['RFF Partial Dependence: ', feature_names{feat}] );
        grid on;

    end


	
	%% Partial Dependence Plots for RFF-informed Mixture-of-GAMs
	feature_names = { ...
		'Acoustic Frequency (Hz)', ...  % plotted on original (exp) scale
		'Angle of Attack', ...
		'Chord Length', ...
		'Free-stream Velocity', ...
		'Suction-side Displacement Thickness' };

	num_plot = 30;     % number of grid points for PDP
	figure_counter = 1;

	for feat = 1:5

		% Create grid in normalized feature space
		x_min = min( X_train_norm( :, feat ) );
		x_max = max( X_train_norm( :, feat ) );
		x_feat_grid = linspace( x_min, x_max, num_plot );

		y_partial = zeros( 1, num_plot );

		for np = 1:num_plot

			% Clamp feature feat to grid value
			X_mod = X_train_norm;
			X_mod( :, feat ) = x_feat_grid( np );

			% Compute RFF reduced representation
			Mix_mat = real( exp( 1i * ( X_mod * omega_used' ) ) .* beta_used' );
			Mix_centered = Mix_mat - h_mean;
			Reduce_dim = Mix_centered * V;
			Reduce_dim_white = Reduce_dim ./ std_B;

			% GMM responsibilities
			Gamma_mat = posterior( gm, Reduce_dim_white );   % (N_train × num_clusters)

			% Predict using each GAM component
			N_train = size( X_train_norm, 1 );
			Y_pred_cluster = zeros( N_train, num_clusters );

			for ell = 1:num_clusters
				total_pred = zeros( N_train, 1 );
				f_list = gam_models_list{ell};

				for j = 1:5
					Phi_j = build_bspline_basis( X_mod( :, j ), f_list{j}.full_knots, 3 );
					total_pred = total_pred + Phi_j * f_list{j}.theta;
				end

				Y_pred_cluster( :, ell ) = alpha_vals( ell ) + total_pred;
			end

			% Mixture prediction, unscale to original y
			y_std = sum( Gamma_mat .* Y_pred_cluster, 2 );
			y_orig = y_std * sigma_y + mu_y;

			% PDP value = mean prediction over all samples
			y_partial( np ) = mean( y_orig );

		end

		% Plot PDP
		figure(); figure_counter = figure_counter + 1;

		% Convert x-axis back to original scale
		x_feat_orig = x_feat_grid * sigma_x( feat ) + mu_x( feat );

		if feat == 1
			% Frequency was log-transformed → convert back
			freq_Hz = exp( x_feat_orig );
			semilogx( freq_Hz, y_partial, 'LineWidth', 2 );
			xlabel( 'Frequency (Hz)  [original scale]' );
		else
			plot( x_feat_orig, y_partial, 'LineWidth', 2 );
			xlabel( feature_names{feat} );
		end

		ylabel( 'Predicted Sound Pressure Level (dB)' );
		title( ['Partial Dependence on ', feature_names{feat}] );
		grid on;

	end
	
	%% === Fit a global GAM on the full normalized training dataset ===
	max_iters = 50;
	smoothing_param = 0.1;
	num_knots = 300;
	degree_gam = 3;
	tol = 1e-3;

	[ alpha_global, f_list_global, ~ ] = ...
		fit_gam_backfitting( X_train_norm, y_train_norm, max_iters, smoothing_param, num_knots, degree_gam, tol );
							 
	% Partial Dependence for Global GAM
	feature_names = { ...
		'Acoustic Frequency (Hz)', ...
		'Angle of Attack', ...
		'Chord Length', ...
		'Free-stream Velocity', ...
		'Suction-side Displacement Thickness' };

	num_plot = 30;
	figure_counter = 1;
	N = size( X_train_norm, 1 );

	for feat = 1:5

		% Create grid in normalized scale
		x_min = min( X_train_norm( :, feat ) );
		x_max = max( X_train_norm( :, feat ) );
		x_feat_grid = linspace( x_min, x_max, num_plot );

		y_pdp = zeros( 1, num_plot );

		for np = 1:num_plot
			% Clamp feature to grid value
			X_mod = X_train_norm;
			X_mod( :, feat ) = x_feat_grid( np );

			% Evaluate the global GAM on X_mod
			y_std = alpha_global * ones( N, 1 );
			for j = 1:5
				Phi_j = build_bspline_basis( X_mod( :, j ), f_list_global{j}.full_knots, degree_gam );
				y_std = y_std + Phi_j * f_list_global{j}.theta;
			end

			% Convert to original scale
			y_orig = y_std * sigma_y + mu_y;

			% Step 4: PDP value
			y_pdp( np ) = mean( y_orig );
		end

		% Plotting

		figure(); figure_counter = figure_counter + 1;

		% Convert normalized x to original scale
		x_feat_orig = x_feat_grid * sigma_x( feat ) + mu_x( feat );

		if feat == 1
			% Frequency was log-transformed → back-transform
			freq_Hz = exp( x_feat_orig );
			semilogx( freq_Hz, y_pdp, 'LineWidth', 2 );
			xlabel( 'Frequency (Hz) [original scale]' );
		else
			plot( x_feat_orig, y_pdp, 'LineWidth', 2 );
			xlabel( feature_names{feat} );
		end

		ylabel( 'Predicted SPL (dB)' );
		title( ['Global GAM Partial Dependence: ', feature_names{feat}] );
		grid on;

	end



	
	
	
	
	
	
	