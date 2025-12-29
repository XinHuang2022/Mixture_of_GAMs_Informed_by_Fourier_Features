    
    clear

    %% load housing.csv
	data_file = fullfile( pwd, 'data', 'housing.csv' );
    Data_table = readtable( data_file );
    Data_array = table2array( Data_table( :, 1 : 9 ) );
	Data_array = Data_array( ~any( isnan( Data_array ), 2 ), : );
	J_no_nan = size( Data_array, 1 );
	% Add path for helper functions
	addpath( genpath( 'utilities' ) );

	
	%% Data processing to obtain mean value of rooms in each area
    ave_num_room = Data_array( :, 4 ) ./ Data_array( :, 7 );
    ave_num_bedroom = Data_array( :, 5 ) ./ Data_array( :, 7 );
    ave_occup = Data_array( :, 6 ) ./ Data_array( :, 7 );
    
    Data_array( :, 4 ) = ave_num_room;
    Data_array( :, 5 ) = ave_num_bedroom;
    Data_array( :, 7 ) = ave_occup;

    J = size( Data_array, 1 );
	
	%% Build spatial coordinates (\xi, \eta) from lon/lat
	lon = Data_array( :, 1 );   % longitude
	lat = Data_array( :, 2 );   % latitude
	
	% Projection to map of california state
	crs = projcrs( 3310 );
	[ xi, eta ] = projfwd( crs, lat, lon );       
	XY = [ xi, eta ];          

    %% Preparation of the data set and implementation details
	use_init_omega_zero = false;    	% If set to true, use zero initial values for all frequencies; otherwise, use standard Gaussian distribution for initial frequencies.
	use_log_transform_y = false;    % If set to true, apply a logarithm transform on the y-training data and perform reverse exponential transform on the regressor; otherwise use original y-data
	use_standard_data = true;    	% If set to true, use standardized training data; otherwise use orginal training data.
	use_early_stopping = true;    	% If set to true, implement an early stopping by inspecting on the validation loss.

	d = 8;
	K = 4000;
    d_xy = 2;
	
	% Determine the initial distribution of frequency parameters
	if( use_init_omega_zero )
		omega_init_spat = zeros( K, d_xy );
	else
		omega_init_spat = randn( K, d_xy );
	end
	omega_sample_spat = omega_init_spat;
	
	% Seperation of training and test dataset
	rng( 126 )
	rand_ind = randperm( J );
    test_ratio = 0.2;
    
    J_train = round( ( 1 - test_ratio ) * J );
    train_idx = rand_ind( 1 : J_train );  
    test_idx = rand_ind( J_train + 1 : J ); 
	
	X_train = Data_array( train_idx, 1 : d );
	y_train_ori = Data_array( train_idx, d + 1 );
	X_test = Data_array( test_idx, 1 : d );
	y_test_ori = Data_array( test_idx, d + 1 );
	if( use_log_transform_y )
		y_train = log( y_train_ori );
		y_test = log( y_test_ori );
	else
		y_train = y_train_ori;
		y_test = y_test_ori;
	end

    fprintf( 'Training set size: %d\n', size( X_train, 1 ) );
	J_test = size( X_test, 1 );
	fprintf( 'Testing set size: %d\n', J_test );
	
	if( use_standard_data )
		% normalization of the data based on the statistics of the training data
		X_train_normal = zeros( J_train, d );
		X_test_normal = zeros( J_test, d );
		mu_x_store = zeros( 1, d );
        sigma_x_store = zeros( 1, d );
		for i = 1 : 1 : d
			X_feature_i = X_train( :, i );
			mu_x_i = mean( X_feature_i );
			sigma_x_i = std( X_feature_i );
			X_train_normal( :, i ) = ( X_feature_i - mu_x_i ) / sigma_x_i;
			X_test_feauture_i = X_test( :, i );
			X_test_normal( :, i ) = ( X_test_feauture_i - mu_x_i ) / sigma_x_i;
            mu_x_store( 1, i ) = mu_x_i;
            sigma_x_store( 1, i ) = sigma_x_i;
		end
		
		mu_y = mean( y_train );
		sigma_y = std( y_train );
		y_train_normal = ( y_train - mu_y ) / sigma_y;
	end					
	x_data_train = X_train_normal;
	y_data_train = y_train_normal;
    y_data_train_ori = y_train_ori;
	x_data_test = X_test_normal;
	y_data_test_ori = y_test_ori;
	
	if( use_early_stopping )
		patience = 30;
		beta_best = [];
        omega_best = [];
		best_loss = inf;
		no_improve_count = 0;
    end

    %% Simple normalization of the converted (\xi, \eta) data
    XY_train = XY( train_idx, : );
    XY_train_mean = mean( XY_train, 1 );
    XY_train_std = std( XY_train, 0, 1 );
    XY_train_normal = ( XY_train - XY_train_mean ) ./ XY_train_std;
    
    XY_test = XY( test_idx, : );
    XY_test_normal = ( XY_test - XY_train_mean ) ./ XY_train_std;
    
	%%  Implementation of the resampling algorithm with RFF on spatial coordinates
	num_resample = 300;
	delta = 0.3;
	lambda = K * sqrt( J_train ) / 100;
	Rel_Tol = 1 * 10^( -3 ); 
    epsilon = 0;
	C_mat_init = eye( d_xy );
	C_mat_sample = C_mat_init;
	epsilon_hat = 1e-3;
	
	[ omega_used_spat, beta_used_spat, RMSE_test, RMSE_train, ci95 ] = ...
    rff_resampling_fit( XY_train_normal, y_data_train, XY_test_normal, y_data_test_ori, ...
    sigma_y, mu_y, J_train, J_test, K, ...
    omega_sample_spat, use_standard_data, use_log_transform_y, ...
    use_early_stopping, patience, ...
    num_resample, delta, lambda, Rel_Tol, epsilon, epsilon_hat );


	%% Test implementing the GMM using simple spatial frequencies
	% d_values = [ 2, 3, 4, 5, 6, 7, 8 ];          % PCA dimensions (columns)
	% L_values = [ 3, 4, 5, 6, 7, 8 ];       % Number of clusters (rows)
    d_values = [ 2, 3 ];          % PCA dimensions (columns)
	L_values = [ 7, 8 ];       % Number of clusters (rows)
	num_d = length( d_values );
	num_L = length( L_values );

	RMSE_grid_spat = zeros( num_L, num_d );       % Initialize RMSE storage
	RMSE_CI_upp_spat = zeros( num_L, num_d );
	RMSE_CI_low_spat = zeros( num_L, num_d );
	RMSE_grid_train_spat = zeros( num_L, num_d );
	
	best_RMSE = Inf;
	best_gm = [];
	best_gam_models_list = [];
	best_alpha_vals = [];
	best_V = [];
	best_h_mean = [];
	best_std_B = [];
    
	for i_a = 1 : 1 : num_L
		for i_b = 1 : 1 : num_d
			
			d_hat = d_values( 1, i_b );  % reduced PCA dimension
			num_clusters = L_values( 1, i_a );      % current number of clusters

			Mix_mat_A = real( exp( 1i * ( XY_train_normal * omega_used_spat' ) ) .* beta_used_spat' );  % J_train x K
			h_mean = mean( Mix_mat_A, 1 );
			[ U_mat, Sigma_mat, V_mat ] = svds( Mix_mat_A - h_mean, d_hat );
			Reduce_dim_mat_B = ( Mix_mat_A - h_mean ) * V_mat;
			Reduce_dim_mat_B_white = Reduce_dim_mat_B ./ std( Reduce_dim_mat_B, 0, 1 );

			options = statset( 'Display', 'final', 'MaxIter', 1000, 'TolFun', 1e-6 );
			% num_clusters = 6;
			gm = fitgmdist( Reduce_dim_mat_B_white, num_clusters, ...
			   'Options', options, 'CovarianceType', 'diagonal', 'RegularizationValue', 1e-3, ...
			   'Replicates', 20, 'SharedCovariance', false, 'Start', 'plus' );

			idx = cluster( gm, Reduce_dim_mat_B_white );

			disp( 'Mixture Proportions:' );
			disp( gm.ComponentProportion );    % Row vector of size 1×num_cluster
	
			% Train GAMs per hard cluster (use non-spatial features)
			gam_models_list = cell( num_clusters, 1 );
			alpha_vals = zeros( 1, num_clusters );
			
			lambda_gam = 1e0;
			max_iter = 300;
			knots_per_dim = 30;
			rel_tol = 1e-3;
			degree_gam = 3;                       % cubic splines

			for ell = 1 : 1 : num_clusters
				idx_ell = find( idx == ell );
				X_ell = x_data_train( idx_ell, : );
				y_ell = y_data_train( idx_ell );
				
				% Fit GAM on data from cluster ell
				[ alpha_ell, f_list_ell, delta_ell ] = fit_gam_backfitting( X_ell, y_ell, knots_per_dim, lambda_gam, max_iter, degree_gam, rel_tol );

				% Store the result
				gam_models_list{ell} = f_list_ell;  % Each entry is a cell array of structs
				alpha_vals( 1, ell )  = alpha_ell;
			end
			
			%  Evaluate the test error of the Mixture-GAM model
			Mix_mat_test = real( exp( 1i * ( XY_test_normal * omega_used_spat' ) ) .* beta_used_spat' );  % J_test × K
			% PCA projection
			Mix_centered_test = Mix_mat_test - h_mean;
			Reduce_dim_test = Mix_centered_test * V_mat;  % J_test × d_hat
			Reduce_dim_test_white = Reduce_dim_test ./ std( Reduce_dim_mat_B, 0, 1 );
			
			% Get GMM responsibilities for the test data
			Gamma_mat = posterior( gm, Reduce_dim_test_white );  % J_test × num_clusters
			
			% Predict from each GAM model
			Y_preds_std_test = zeros( J_test, num_clusters );  % Standardized preds
			
			for ell = 1 : 1 : num_clusters
				total_pred_test = zeros( J_test, 1 );
				% Predict on standardized test data using model ell
				f_list_ell = gam_models_list{ell};
				for feature_index = 1 : 1 : d
					x_feature_vals = x_data_test( :, feature_index );
					Phi_feature_index = build_bspline_basis( x_feature_vals, f_list_ell{feature_index}.full_knots, degree_gam );
					total_pred_test = total_pred_test + Phi_feature_index * f_list_ell{feature_index}.theta;
				end
				alpha_ell = alpha_vals( 1, ell );
				Y_preds_std_test( :, ell ) = alpha_ell + total_pred_test;
			end
			
			% Mixture prediction in standardized space
			y_hat_std_test = sum( Gamma_mat .* Y_preds_std_test, 2 );  % J_test × 1
			% Inverse transform back to original scale
			y_hat_test = y_hat_std_test * sigma_y + mu_y;
			
			% Compute RMSE
			RMSE_MGAM = sqrt( mean( ( y_hat_test - y_data_test_ori ).^2 ) );
			fprintf( 'Test root mean squared error for MGAM model: %6.5e\n', RMSE_MGAM );
			
			if ( ( i_a == 1 && i_b == 1 ) || RMSE_MGAM < best_RMSE )
				best_RMSE = RMSE_MGAM;
				best_L = num_clusters;
				best_d = d_hat;
				
				% Store best models and transformation
				best_gm = gm;
				best_idx = idx;
				best_gam_models_list = gam_models_list;
				best_alpha_vals = alpha_vals;
				
				best_V = V_mat;
				best_h_mean = h_mean;
				best_std_B = std( Reduce_dim_mat_B, 0, 1 );
			end

			% Bootstrap
			B_num = 1000;
			rmse_boot = zeros( B_num, 1 );
			resid = y_data_test_ori - y_hat_test;
			
			for b = 1 : 1 : B_num
				idx_b = randi( J_test, J_test, 1 );          % sample indices with replacement
				resamp_resid = resid( idx_b );     % bootstrap sample
				rmse_boot( b ) = sqrt( mean( resamp_resid.^2 ) );
			end
			
			% Confidence interval
			ci95 = quantile( rmse_boot, [ 0.025, 0.975 ] );
			
			fprintf( 'Test RMSE = %.3f, 95%% CI [%.3f, %.3f]\n', RMSE_MGAM, ci95( 1 ), ci95( 2 ) );
			
			RMSE_grid_spat( i_a, i_b ) = RMSE_MGAM;
			RMSE_CI_low_spat( i_a, i_b ) = ci95( 1 );
			RMSE_CI_upp_spat( i_a, i_b ) = ci95( 2 );


			%  Evaluate the training error of the Mixture-GAM model
	
			Mix_mat_train = real( exp( 1i * ( XY_train_normal * omega_used_spat' ) ) .* beta_used_spat' );  % J_test × K
			% PCA projection
			Mix_centered_train = Mix_mat_train - h_mean;
			Reduce_dim_train = Mix_centered_train * V_mat;  % J_test × d_hat
			Reduce_dim_train_white = Reduce_dim_train ./ std( Reduce_dim_mat_B, 0, 1 );
			
			% Get GMM responsibilities for the training data
			Gamma_mat_train = posterior( gm, Reduce_dim_train_white );  % J_test × num_clusters
			
			% Predict from each GAM model
			Y_preds_std_train = zeros( J_train, num_clusters );  % Standardized preds
			
			for ell = 1 : 1 : num_clusters
				total_pred_train = zeros( J_train, 1 );
				% Predict on standardized test data using model ell
				f_list_ell = gam_models_list{ell};
				for feature_index = 1 : 1 : d
					x_feature_vals = x_data_train( :, feature_index );
					Phi_feature_index = build_bspline_basis( x_feature_vals, f_list_ell{feature_index}.full_knots, degree_gam );
					total_pred_train = total_pred_train + Phi_feature_index * f_list_ell{feature_index}.theta;
				end
				alpha_ell = alpha_vals( 1, ell );
				Y_preds_std_train( :, ell ) = alpha_ell + total_pred_train;
			end
			
			% Mixture prediction in standardized space
			y_hat_std_train = sum( Gamma_mat_train .* Y_preds_std_train, 2 );  % J_test × 1
			% Inverse transform back to original scale
			y_hat_train = y_hat_std_train * sigma_y + mu_y;
			
			% Compute RMSE
			RMSE_MGAM_train = sqrt( mean( ( y_hat_train - y_data_train_ori ).^2 ) );
			fprintf( 'Training root mean squared error for MGAM model: %6.5e\n', RMSE_MGAM_train );
			RMSE_grid_train_spat( i_a, i_b ) = RMSE_MGAM_train;
			
		end
    end

    figure();
    imagesc( d_values, L_values, RMSE_grid_spat / 1e5 );   % rows = L, cols = d
    colorbar;
    colormap( parula );                        
    
    xlabel( 'PCA dimension $d$' );
    ylabel( 'Number of GMM clusters $L$' );
    title( 'Test RMSE of MGAM' );

    % Annotate each cell with RMSE
    for i_a = 1 : 1 : num_L
        for i_b = 1 : 1 : num_d
            rmse_val = RMSE_grid_spat( i_a, i_b );
            if ~isnan( rmse_val )
                text( d_values( i_b ), L_values( i_a ), sprintf( '%.3f', rmse_val / 1e5 ), ...
                    'HorizontalAlignment', 'center', ...
                    'Color', 'w', 'FontSize', 10 );
            end
        end
    end
	
	%% Plot the clustered data points.
    % Define custom transparent colors
    color_list = {
        [ 0.2, 0.5, 1.0, 0.3 ],  % light transparent blue (RGBA)
        [ 1.0, 0.5, 0.1, 0.3 ]   % light transparent orange
    };
    states = shaperead( 'usastatehi', 'UseGeoCoords', true );
    CA = states( strcmp( {states.Name}, 'California' ) );
    
    % Convert CA outline from lat/lon to (xi, eta)
    [ cal_xi, cal_eta ] = projfwd( crs, [CA.Lat], [CA.Lon] );

    for ell = 1 : num_clusters
        idx_ell = find( best_idx == ell );
    
        figure();
    
        % Choose color based on comparison strategy: blue for one, orange for another
        color_idx = 1;  % or 2
        rgba = color_list{color_idx};
    
        % Plot scatter with transparency
        scatter( XY_train( idx_ell, 1 ), XY_train( idx_ell, 2 ), ...
                 8, 'filled', ...
                 'MarkerFaceColor', rgba(1:3), ...
                 'MarkerFaceAlpha', rgba(4) );  % transparent fill
    
        hold on;
    
        % Overlay California outline with thinner line (e.g., linewidth 0.8)
        plot( cal_xi, cal_eta, 'k-', 'LineWidth', 0.8 );
    
        title( sprintf( 'Cluster %d using spatial RFF', ell ) );
        xlabel( '$x$' );
        ylabel( '$y$' )
        axis equal;
        grid on;
    end
	
	
    %% Plot the new version of empirical histogram with compass
    density = ksdensity( omega_used_spat, omega_used_spat );  % local densities
	th = prctile( density, 50 );  % top 50% densest points
	Omega_core = omega_used_spat( density > th, : );

	[ ~,~,V_core ] = svd( Omega_core - mean( Omega_core, 1 ),'econ' );
	v1_core = V_core( :, 1 );
	v1 = -v1_core / norm( v1_core );
	v2_core = V_core( :, 2 );
	v2 = v2_core / norm( v2_core );	
	
	% --- Compute histogram ---
	num_bins = 200;
	[ H, xedges, yedges ] = histcounts2( omega_used_spat( :, 1 ), omega_used_spat( :, 2 ), num_bins );
	H_normalized = H / sum( H(:) );
    % Compute bin centers (instead of edges)
    x_centers = ( xedges( 1 : end - 1 ) + xedges( 2 : end ) ) / 2;
    y_centers = ( yedges( 1 : end - 1 ) + yedges( 2 : end ) ) / 2;
    [ X, Y ] = meshgrid( x_centers, y_centers );
    
    % Use half the range of the histogram domain
	len = 0.5 * max( range( xedges ), range( yedges ) );

    threshold = 0;  
    H_thresholded = H_normalized;
    H_thresholded( H_thresholded < threshold ) = 0;
    
    % === Plot using SURF ===
    figure;
    surf( X, Y, H_thresholded', 'EdgeColor', 'none' );
    view( 2 );  % top-down view (like heatmap)
    axis equal tight;
    xlabel( '$\omega_{\xi}$', 'Interpreter', 'latex' );
    ylabel( '$\omega_{\eta}$', 'Interpreter', 'latex' );
    title( '2D Histogram of Frequencies' );
    colormap( parula );
    colorbar;
    hold on

    % Create a small inset axes (normalized coordinates of the figure window)
    ax_compass = axes( 'Position',[ 0.52 0.72 0.15 0.15 ] ); 
    hold( ax_compass,'on' );
    axis( ax_compass,'equal' );
    axis( ax_compass,'off' );   % hide ticks

    scale = 0.8;  % arrow length inside inset
    lw = 2;       % line width
    
    % Plot v1
    quiver( ax_compass, 0, 0,  0.4 * scale*v1(1),  0.4 * scale*v1(2), 0, ...
           'Color',[0.4, 0.55, 0.2],'LineWidth',lw,'MaxHeadSize',0 );
    
    quiver( ax_compass, 0, 0, -scale*v1(1), -scale*v1(2), 0, ...
           'Color',[0.4, 0.55, 0.2],'LineWidth',lw,'MaxHeadSize',0.8 );
    
    % Plot v2
    quiver( ax_compass, 0, 0,  0.4 * scale*v2(1),  0.4 * scale*v2(2), 0, ...
           'Color',[0.4, 0.55, 0.2],'LineWidth',lw,'MaxHeadSize',0 );
    
    quiver( ax_compass, 0, 0, -0.45 * scale*v2(1), -0.45 * scale*v2(2), 0, ...
           'Color',[0.4, 0.55, 0.2],'LineWidth',lw,'MaxHeadSize',0.8 );
    
    % Labels
    text( scale*1.1*v1(1), scale*1.1*v1(2), '$v_1$', ...
         'Interpreter','latex','Color',[0.4, 0.55, 0.2],'FontSize',16,'Parent',ax_compass );
    
    text( scale*1.1*v2(1), scale*1.1*v2(2), '$v_2$', ...
         'Interpreter','latex','Color',[0.4, 0.55, 0.2],'FontSize',16,'Parent',ax_compass );

    %% Plot the new version of the price data points with compass
    figure;
    hold on;

    states = shaperead( 'usastatehi', 'UseGeoCoords', true );
    CA = states( strcmp( {states.Name}, 'California' ) );
    
    % Convert CA outline from lat/lon to (xi, eta)
    [ cal_xi, cal_eta ] = projfwd( crs, [CA.Lat], [CA.Lon] );
    % Overlay California outline
    plot( cal_xi, cal_eta, 'k-', 'LineWidth', 1.5 );
    axis equal;
    grid on;
    
    % Plot the California boundary (already in projected (xi, eta) coords)
    plot( cal_xi, cal_eta, 'k-', 'LineWidth', 1.5 );
    
    % Scatter plot of housing data
    price = Data_array( :, 9 );      % housing prices
    
    % Scatter with color mapping
    scatter( xi, eta, 25, price, 'filled' ); 
    colormap( parula );   
    colorbar;

    % Axis formatting
    axis equal;
    axis tight;
    xlabel( '$\xi$', 'Interpreter', 'latex', 'FontSize', 14 );
    ylabel( '$\eta$', 'Interpreter', 'latex', 'FontSize', 14 );
    title( 'California Housing Prices with State Outline', 'Interpreter', 'latex', 'FontSize', 14 );
    
    % Improve aesthetics
    set( gca, 'FontSize', 12 );
    grid off
    ax = gca;
    ax.YAxisLocation = 'left'; 
    ax.YRuler.SecondaryLabel.VerticalAlignment = 'bottom';
    ax.YRuler.SecondaryLabel.HorizontalAlignment = 'left';

    % Create a small inset axes (normalized coordinates of the figure window)
    ax_compass = axes( 'Position', [ 0.52 0.66 0.18 0.18 ] ); 
    hold( ax_compass, 'on' );
    axis( ax_compass, 'equal' );
    axis( ax_compass, 'off' );   % hide ticks

    scale = 0.8;  % arrow length inside inset
    lw = 2;       % line width
    
    % Plot v1
    quiver( ax_compass, 0, 0,  0.4 * scale*v1(1),  0.4 * scale*v1(2), 0, ...
           'Color',[0.4, 0.55, 0.2],'LineWidth',lw,'MaxHeadSize',0 );
    
    quiver( ax_compass, 0, 0, -scale*v1(1), -scale*v1(2), 0, ...
           'Color',[0.4, 0.55, 0.2],'LineWidth',lw,'MaxHeadSize',0.8 );
    
    % Plot v2 
    quiver( ax_compass, 0, 0,  0.4 * scale*v2(1),  0.4 * scale*v2(2), 0, ...
           'Color',[0.4, 0.55, 0.2],'LineWidth',lw,'MaxHeadSize',0 );
    
    quiver( ax_compass, 0, 0, -0.45 * scale*v2(1), -0.45 * scale*v2(2), 0, ...
           'Color',[0.4, 0.55, 0.2],'LineWidth',lw,'MaxHeadSize',0.8 );
    
    % Labels
    text( scale*1.1*v1(1), scale*1.1*v1(2), '$v_1$', ...
         'Interpreter','latex','Color',[0.4, 0.55, 0.2],'FontSize',16,'Parent',ax_compass );
    
    text( scale*1.1*v2(1), scale*1.1*v2(2), '$v_2$', ...
         'Interpreter','latex','Color',[0.4, 0.55, 0.2],'FontSize',16,'Parent',ax_compass );

    %% Implementation of RFF model on the eight covariates
    % Determine the initial distribution of frequency parameters
	if( use_init_omega_zero )
		omega_init = zeros( K, d );
	else
		omega_init = randn( K, d );
	end
	omega_sample = omega_init;
    if( use_early_stopping )
		patience = 30;
		beta_best = [];
        omega_best = [];
		best_loss = inf;
		no_improve_count = 0;
    end

    num_resample = 300;
	delta = 0.3;
	lambda = K * sqrt( J_train ) / 100;
	Rel_Tol = 1 * 10^( -3 ); 
    % epsilon = K^(-0.5) / 200;
    epsilon = 0;
	C_mat_init = eye( d );
	C_mat_sample = C_mat_init;
	epsilon_hat = 1e-3;

    [ omega_used_comp, beta_used_comp, RMSE_test, RMSE_train, ci95_comp ] = ...
    rff_resampling_fit( x_data_train, y_data_train, x_data_test, y_data_test_ori, ...
    sigma_y, mu_y, J_train, J_test, K, ...
    omega_sample, use_standard_data, use_log_transform_y, ...
    use_early_stopping, patience, ...
    num_resample, delta, lambda, Rel_Tol, epsilon, epsilon_hat );
	
	
    %% Visualization of the partial dependence on each covariate dimension
	feature_names = { 'Longitude', 'Latitude', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'MedInc' };
    for i = 1 : 1 : d
        num_plot_point = 30;
        x_i_plot_vec = linspace( min( x_data_train( :, i ) ), max( x_data_train( :, i ) ), num_plot_point );
		y_i_plot_vec = zeros( 1, num_plot_point );
		for np = 1 : 1 : num_plot_point
			x_data_train_mat_i = x_data_train;
			x_data_train_mat_i( :, i ) = x_i_plot_vec( 1, np ) * ones( J_train, 1 );
			y_i_plot_xi = mean( real( exp( 1i * ( x_data_train_mat_i * omega_used_comp' ) ) * beta_used_comp ) );
			y_i_plot_vec( 1, np ) = y_i_plot_xi * sigma_y + mu_y;
		end
		figure;
        sigma_x_i = sigma_x_store( 1, i );
        mu_x_i = mu_x_store( 1, i );
   
		plot( x_i_plot_vec * sigma_x_i + mu_x_i, y_i_plot_vec )
		xlabel( [ feature_names{i}, '(original scale)' ] );
        ylabel( 'Predicted y (original scale)' );
        title( [ 'Partial Dependence on ', feature_names{i} ] );
    end

    %% Grid search for optimal hyperparameters L and d
    % d_values = [ 2, 3, 4, 5, 6, 7, 8 ];          % PCA dimensions (columns)
	% L_values = [ 3, 4, 5, 6, 7, 8 ];       % Number of clusters (rows)
    d_values = [ 2, 3 ];          % PCA dimensions (columns)
	L_values = [ 7, 8 ];       % Number of clusters (rows)
	num_d = length( d_values );
	num_L = length( L_values );

	RMSE_grid = zeros( num_L, num_d );       % Initialize RMSE storage
	RMSE_CI_upp = zeros( num_L, num_d );
	RMSE_CI_low = zeros( num_L, num_d );
	RMSE_grid_train = zeros( num_L, num_d );
	
	best_RMSE = Inf;
	best_gm = [];
	best_gam_models_list = [];
	best_alpha_vals = [];
	best_V = [];
	best_h_mean = [];
	best_std_B = [];
	
	for i_a = 1 : 1 : num_L
		for i_b = 1 : 1 : num_d
			
			d_hat = d_values( 1, i_b );  % reduced PCA dimension
			num_clusters = L_values( 1, i_a );      % current number of clusters

			Mix_mat_A = real( exp( 1i * ( x_data_train * omega_used_comp' ) ) .* beta_used_comp' );  % Ntr x K
			% d_hat = 5;
			h_mean = mean( Mix_mat_A, 1 );
			[ U_mat, Sigma_mat, V_mat ] = svds( Mix_mat_A - h_mean, d_hat );
			Reduce_dim_mat_B = ( Mix_mat_A - h_mean ) * V_mat;
			Reduce_dim_mat_B_white = Reduce_dim_mat_B ./ std( Reduce_dim_mat_B, 0, 1 );

			options = statset( 'Display', 'final', 'MaxIter', 1000, 'TolFun', 1e-6 );
			% num_clusters = 6;
			gm = fitgmdist( Reduce_dim_mat_B_white, num_clusters, ...
			   'Options', options, 'CovarianceType', 'diagonal', 'RegularizationValue', 1e-3, ...
			   'Replicates', 20, 'SharedCovariance', false, 'Start', 'plus' );

			idx = cluster( gm, Reduce_dim_mat_B_white );

			disp( 'Mixture Proportions:' );
			disp( gm.ComponentProportion );    % Row vector of size 1×num_cluster
	
	
			% Train GAMs per hard cluster (use non-spatial features)
			gam_models_list = cell( num_clusters, 1 );
			alpha_vals = zeros( 1, num_clusters );
			
			lambda_gam = 1e0;
			max_iter = 300;
			knots_per_dim = 30;
			rel_tol = 1e-3;
			degree_gam = 3;                       % cubic splines

			for ell = 1 : 1 : num_clusters
				idx_ell = find( idx == ell );
				X_ell = x_data_train( idx_ell, : );
				y_ell = y_data_train( idx_ell );
				
				% Fit GAM on data from cluster ell
				[ alpha_ell, f_list_ell, delta_ell ] = fit_gam_backfitting( X_ell, y_ell, knots_per_dim, lambda_gam, max_iter, degree_gam, rel_tol );

				% Store the result
				gam_models_list{ell} = f_list_ell;  % Each entry is a cell array of structs
				alpha_vals( 1, ell )  = alpha_ell;
			end
			
			%  Evaluate the test error of the Mixture-GAM model
			Mix_mat_test = real( exp( 1i * ( x_data_test * omega_used_comp' ) ) .* beta_used_comp' );  % J_test × K
			% PCA projection
			Mix_centered_test = Mix_mat_test - h_mean;
			Reduce_dim_test = Mix_centered_test * V_mat;  % J_test × d_hat
			Reduce_dim_test_white = Reduce_dim_test ./ std( Reduce_dim_mat_B, 0, 1 );
			
			% Get GMM responsibilities for the test data
			Gamma_mat = posterior( gm, Reduce_dim_test_white );  % J_test × num_clusters
			
			% Predict from each GAM model
			Y_preds_std_test = zeros( J_test, num_clusters );  % Standardized preds
			
			for ell = 1 : 1 : num_clusters
				total_pred_test = zeros( J_test, 1 );
				% Predict on standardized test data using model ell
				f_list_ell = gam_models_list{ell};
				for feature_index = 1 : 1 : d
					x_feature_vals = x_data_test( :, feature_index );
					Phi_feature_index = build_bspline_basis( x_feature_vals, f_list_ell{feature_index}.full_knots, degree_gam );
					total_pred_test = total_pred_test + Phi_feature_index * f_list_ell{feature_index}.theta;
				end
				alpha_ell = alpha_vals( 1, ell );
				Y_preds_std_test( :, ell ) = alpha_ell + total_pred_test;
			end
			
			% Mixture prediction in standardized space
			y_hat_std_test = sum( Gamma_mat .* Y_preds_std_test, 2 );  % J_test × 1
			% Inverse transform back to original scale
			y_hat_test = y_hat_std_test * sigma_y + mu_y;
			
			% Compute RMSE
			RMSE_MGAM = sqrt( mean( ( y_hat_test - y_data_test_ori ).^2 ) );
			fprintf( 'Test root mean squared error for MGAM model: %6.5e\n', RMSE_MGAM );
			
			if( ( i_a == 1 && i_b == 1 ) || RMSE_MGAM < best_RMSE )
				best_RMSE = RMSE_MGAM;
				best_L = num_clusters;
				best_d = d_hat;
				
				% Store best models and transformation
				best_gm = gm;
				best_idx = idx;
				best_gam_models_list = gam_models_list;
				best_alpha_vals = alpha_vals;
				
				best_V = V_mat;
				best_h_mean = h_mean;
				best_std_B = std( Reduce_dim_mat_B, 0, 1 );
			end

			% Bootstrap
			B_num = 1000;
			rmse_boot = zeros( B_num, 1 );
			resid = y_data_test_ori - y_hat_test;
			
			for b = 1 : 1 : B_num
				idx_b = randi( J_test, J_test, 1 );          % sample indices with replacement
				resamp_resid = resid( idx_b );     % bootstrap sample
				rmse_boot( b ) = sqrt( mean( resamp_resid.^2 ) );
			end
			
			% Confidence interval
			ci95 = quantile( rmse_boot, [ 0.025, 0.975 ] );
			
			fprintf( 'Test RMSE = %.3f, 95%% CI [%.3f, %.3f]\n', RMSE_MGAM, ci95( 1 ), ci95( 2 ) );
			
			RMSE_grid( i_a, i_b ) = RMSE_MGAM;
			RMSE_CI_low( i_a, i_b ) = ci95( 1 );
			RMSE_CI_upp( i_a, i_b ) = ci95( 2 );


			%  Evaluate the training error of the Mixture-GAM model
			
			Mix_mat_train = real( exp( 1i * ( x_data_train * omega_used_comp' ) ) .* beta_used_comp' );  % J_test × K
			% PCA projection
			Mix_centered_train = Mix_mat_train - h_mean;
			Reduce_dim_train = Mix_centered_train * V_mat;  % J_test × d_hat
			Reduce_dim_train_white = Reduce_dim_train ./ std( Reduce_dim_mat_B, 0, 1 );
			
			% Get GMM responsibilities for the training data
			Gamma_mat_train = posterior( gm, Reduce_dim_train_white );  % J_test × num_clusters
			
			% Predict from each GAM model
			Y_preds_std_train = zeros( J_train, num_clusters );  % Standardized preds
			
			for ell = 1 : 1 : num_clusters
				total_pred_train = zeros( J_train, 1 );
				% Predict on standardized test data using model ell
				f_list_ell = gam_models_list{ell};
				for feature_index = 1 : 1 : d
					x_feature_vals = x_data_train( :, feature_index );
					Phi_feature_index = build_bspline_basis( x_feature_vals, f_list_ell{feature_index}.full_knots, degree_gam );
					total_pred_train = total_pred_train + Phi_feature_index * f_list_ell{feature_index}.theta;
				end
				alpha_ell = alpha_vals( 1, ell );
				Y_preds_std_train( :, ell ) = alpha_ell + total_pred_train;
			end
			
			% Mixture prediction in standardized space
			y_hat_std_train = sum( Gamma_mat_train .* Y_preds_std_train, 2 );  % J_test × 1
			% Inverse transform back to original scale
			y_hat_train = y_hat_std_train * sigma_y + mu_y;
			
			% Compute RMSE
			RMSE_MGAM_train = sqrt( mean( ( y_hat_train - y_data_train_ori ).^2 ) );
			fprintf( 'Training root mean squared error for MGAM model: %6.5e\n', RMSE_MGAM_train );
			RMSE_grid_train( i_a, i_b ) = RMSE_MGAM_train;
			
		end
    end

    figure();
    imagesc( d_values, L_values, RMSE_grid / 1e5 );   % rows = L, cols = d
    colorbar;
    colormap( parula );                         % or try 'hot', 'cool', etc.
    
    xlabel( 'PCA dimension $d$' );
    ylabel( 'Number of GMM clusters $L$' );
    title( 'Test RMSE of MGAM' );
	
	%% Plot the 2d-distribution of each cluster component's position
    color_list = {
        [0.2, 0.5, 1.0, 0.3],  % light transparent blue (RGBA)
        [1.0, 0.5, 0.1, 0.1]   % light transparent orange
    };

    states = shaperead( 'usastatehi', 'UseGeoCoords', true );
    CA = states( strcmp( {states.Name}, 'California' ) );
    
    % Convert CA outline from lat/lon to (xi, eta)
    [ cal_xi, cal_eta ] = projfwd( crs, [CA.Lat], [CA.Lon] );

    for ell = 1 : num_clusters
        idx_ell = find( best_idx == ell );
    
        figure();
    
        % Choose color based on comparison strategy: blue for one, orange for another
        color_idx = 2;  % or 2
        rgba = color_list{color_idx};
    
        % Plot scatter with transparency
        scatter( XY_train( idx_ell, 1 ), XY_train( idx_ell, 2 ), ...
                 8, 'filled', ...
                 'MarkerFaceColor', rgba(1:3), ...
                 'MarkerFaceAlpha', rgba(4) );  % transparent fill
    
        hold on;
    
        % Overlay California outline with thinner line (e.g., linewidth 0.8)
        plot( cal_xi, cal_eta, 'k-', 'LineWidth', 0.8 );
    
        title(sprintf( 'Cluster %d, complete RFF', ell ) );
        xlabel( '$x$' );
        ylabel( '$y$' );
        axis equal;
        grid on;
    end
	
	
	%% Partial Dependence for trained Mixture-of-GAMs (MGAM)
	feature_names = { 'Longitude', 'Latitude', 'HouseAge', 'AveRooms', ...
					  'AveBedrms', 'Population', 'AveOccup', 'MedInc' };

	num_plot_point = 30;
	figure_counter = 1;

	for feat = 1:d

		% Create evaluation grid for feature 'feat'
		x_feat_vals = linspace( min( x_data_train( :, feat ) ), ...
								max( x_data_train( :, feat ) ), num_plot_point );
		y_partial_vals = zeros( 1, num_plot_point );

		for np = 1:num_plot_point
			
			% Construct modified dataset
			x_mod = x_data_train;
			x_mod( :, feat ) = x_feat_vals( np );

			% Recompute RFF-based mixture representation
			Mix_mat = real( exp( 1i * ( x_mod * omega_used_comp' ) ) .* beta_used_comp' );
			Mix_centered = Mix_mat - h_mean;
			Reduce_dim = Mix_centered * V_mat;
			Reduce_dim_white = Reduce_dim ./ std( Reduce_dim_mat_B, 0, 1 );

			% Compute GMM responsibilities
			Gamma_mat = posterior( gm, Reduce_dim_white );   % (J_train × num_clusters)

			% Predict with each GAM component
			Y_pred_all = zeros( J_train, num_clusters );  % standardized scale

			for ell = 1:num_clusters
				total_pred = zeros( J_train, 1 );
				f_list = gam_models_list{ell};

				for j = 1:d
					x_feature_vals = x_mod( :, j );
					Phi_j = build_bspline_basis( ...
								x_feature_vals, ...
								f_list{j}.full_knots, ...
								degree_gam);
					total_pred = total_pred + Phi_j * f_list{j}.theta;
				end

				alpha_ell = alpha_vals(1, ell);
				Y_pred_all(:, ell) = alpha_ell + total_pred;
			end

			% Final MGAM mixture prediction
			y_std = sum( Gamma_mat .* Y_pred_all, 2 );     % standardized
			y_orig = y_std * sigma_y + mu_y;             % original scale

			% PDP value = mean prediction
			y_partial_vals( np ) = mean( y_orig );

		end

        % Plot PDP for this feature
        figure; figure_counter = figure_counter + 1;
    
        % Transform x-axis to original scale
        x_feat_vals_orig = x_feat_vals * sigma_x_store( feat ) + mu_x_store( feat );
    
        plot( x_feat_vals_orig, y_partial_vals, 'LineWidth', 2 );
        xlabel( [feature_names{feat}, ' (original scale)'] );
        ylabel( 'Predicted y (original scale)' );
        title( ['MGAM Partial Dependence: ', feature_names{feat}] );
        grid on;
    
    end


	%% LASSO with 10-fold CV on the training set
    feature_names = { 'Longitude', 'Latitude', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'MedInc' };
	lambdaVec = logspace( -6, 3, 60 );
	[ Beta, FitInfo ] = lasso( x_data_train, y_data_train, 'CV', 10, 'Lambda', lambdaVec, 'Standardize', false, 'Alpha', 1, 'NumLambda', 100 );           

	% Pick the lambda that minimizes CV MSE (or use the 1-SE rule for sparsity)
	idxMinMSE = FitInfo.IndexMinMSE;
	idx1SE    = FitInfo.Index1SE;

	betaMin   = Beta( :, idxMinMSE );
	b0Min     = FitInfo.Intercept( idxMinMSE );

	beta1SE   = Beta( :, idx1SE );
	b01SE     = FitInfo.Intercept( idx1SE );

	% Evaluate on the held-out test set
	yhat_min = x_data_test * betaMin + b0Min;
	yhat_min_rescale = yhat_min * sigma_y + mu_y;
	rmse_min = sqrt( mean( ( y_data_test_ori - yhat_min_rescale ).^2 ) );

	yhat_1se = x_data_test * beta1SE + b01SE;
	yhat_1se_rescale = yhat_1se * sigma_y + mu_y;
	rmse_1se = sqrt( mean( ( y_data_test_ori - yhat_1se_rescale ).^2 ) );

	fprintf( 'RMSE (min-CV-MSE lambda): %.4f\n', rmse_min );
	fprintf( 'RMSE (1-SE lambda)      : %.4f\n', rmse_1se );
	
	% Bootstrap
    B_num = 1000;
    rmse_boot = zeros( B_num, 1 );
    resid = y_data_test_ori - yhat_min_rescale;
    
    for b = 1 : 1 : B_num
        idx_b = randi( J_test, J_test, 1 );          % sample indices with replacement
        resamp_resid = resid( idx_b );     % bootstrap sample
        rmse_boot( b ) = sqrt( mean( resamp_resid.^2 ) );
    end
    
    % Confidence interval
    ci95 = quantile( rmse_boot, [ 0.025, 0.975 ] );
    
    fprintf( 'Test RMSE = %.3f, 95%% CI [%.3f, %.3f]\n', mean( rmse_boot ), ci95( 1 ), ci95( 2 ) );

	% Inspect sparsity / interpretability
	nz_min = find( betaMin ~= 0 );
	disp( 'Nonzero coefficients (min-CV-MSE):' );
	disp( table( feature_names( nz_min )', betaMin( nz_min ), 'VariableNames', {'Predictor','Coefficient'} ) );

	nz_1se = find( beta1SE ~= 0 );
	disp( 'Nonzero coefficients (1-SE):' );
	disp( table( feature_names( nz_1se )', beta1SE( nz_1se ), 'VariableNames', {'Predictor','Coefficient'} ) );
    
    % Evaluating the training error
    yhat_train = x_data_train * betaMin + b0Min;
	yhat_train_rescale = yhat_train * sigma_y + mu_y;
	rmse_train = sqrt( mean( ( y_data_train_ori - yhat_train_rescale ).^2 ) );
    fprintf( 'Training RMSE: %.5f\n', rmse_train );

    %% Fit the GAM model on the entire training data set and evaluate the test error
	tic;
    knots_per_dim = 30;
    max_iter = 100;
    lambda_gam = 1e-1;
	degree_gam = 3;
    use_ortho_global = false;
	[ alpha, f_list, delta ] = fit_gam_backfitting( x_data_train, y_data_train, knots_per_dim, lambda_gam, max_iter, degree_gam, use_ortho_global );    % fit_gam_backfitting( X, y, knots_per_dim, lambda, max_iter )
	elapsed_time = toc;  % end timer and get elapsed time
    fprintf( 'Fitting completed in %.2f seconds with relative difference %5.4e.\n', elapsed_time, delta );
	
	% Test error
	Y_preds_gam_test = zeros( J_test, 1 );  % Standardized preds
	for feature_index = 1 : 1 : d
		x_feature_vals = x_data_test( :, feature_index );
		% Phi_feature_index = build_bspline_basis( x_feature_vals, f_list{feature_index}.full_knots, degree_gam, use_ortho_global );
		Phi_feature_index = build_bspline_basis( x_feature_vals, f_list{feature_index}.full_knots, degree_gam );
        Y_preds_gam_test = Y_preds_gam_test + Phi_feature_index * f_list{feature_index}.theta;
	end
	Y_preds_gam_test = alpha + Y_preds_gam_test;

	% Inverse transform back to original scale
	y_hat_gam_test = Y_preds_gam_test * sigma_y + mu_y;
	
	% Compute RMSE
	RMSE_GAM = sqrt( mean( ( y_hat_gam_test - y_data_test_ori ).^2 ) );
	fprintf( 'Test root mean squared error for GAM model: %6.5e\n', RMSE_GAM );

    % Training error
	Y_preds_gam_train = zeros( J_train, 1 );  % Standardized preds
	for feature_index = 1 : 1 : d
		x_feature_vals = x_data_train( :, feature_index );
		% Phi_feature_index = build_bspline_basis( x_feature_vals, f_list{feature_index}.full_knots, degree_gam, use_ortho_global );
		Phi_feature_index = build_bspline_basis( x_feature_vals, f_list{feature_index}.full_knots, degree_gam );
        Y_preds_gam_train = Y_preds_gam_train + Phi_feature_index * f_list{feature_index}.theta;
	end
	Y_preds_gam_train = alpha + Y_preds_gam_train;

	% Inverse transform back to original scale
	y_hat_gam_train = Y_preds_gam_train * sigma_y + mu_y;
	
	% Compute RMSE
	RMSE_GAM_train = sqrt( mean( ( y_hat_gam_train - y_data_train_ori ).^2 ) );
	fprintf( 'Training root mean squared error for GAM model: %6.5e\n', RMSE_GAM_train );

    % Bootstrap
	B_num = 1000;
    rmse_boot = zeros( B_num, 1 );
    resid = y_data_test_ori - y_hat_gam_test;
    
    for b = 1 : 1 : B_num
        idx_b = randi( J_test, J_test, 1 );          % sample indices with replacement
        resamp_resid = resid( idx_b );     % bootstrap sample
        rmse_boot( b ) = sqrt( mean( resamp_resid.^2 ) );
    end
    
    % Confidence interval
    ci95 = quantile( rmse_boot, [ 0.025, 0.975 ] );
    
    fprintf( 'Test RMSE = %.3f, 95%% CI [%.3f, %.3f]\n', mean( rmse_boot ), ci95( 1 ), ci95( 2 ) );
	
	%% Plot the partial dependence of the customized fitting of global GAM model
	for feature_index = 1 : 1 : d

		x_vals = linspace( min( x_data_train( :, feature_index ) ), max( x_data_train( :, feature_index ) ), num_plot_point );
		y_vals = zeros( 1, num_plot_point );
    
		tic;
		for np = 1 : 1 : num_plot_point
			x_fixed = x_vals( 1, np );

			% Compute partial prediction across all data points with x_j = x_fixed
			total_pred = zeros( J_train, 1 );

			for j = 1 : 1 : d
				if j == feature_index
					Phi_fixed = build_bspline_basis( x_fixed, f_list{j}.full_knots, degree_gam );  % 1 × nBasis
					f_j_x_fixed = Phi_fixed * f_list{j}.theta;  % scalar
					total_pred = total_pred + f_j_x_fixed * ones( J_train, 1 );
				else
					x_j_eval = x_data_train( :, j );  % original training values for other features
					Phi_j = build_bspline_basis( x_j_eval, f_list{j}.full_knots, degree_gam );  % J × nBasis
					total_pred = total_pred + Phi_j * f_list{j}.theta;
				end
			end
			
			y_vals( 1, np ) = mean( total_pred ) + alpha;  % marginal effect over x_{-j}
		end
		elapsed_time_2 = toc;
		fprintf( 'Plotting partial dependence for feature %d completed.\n', feature_index );

		% Reverse standardization for x and y
		x_plot = x_vals * sigma_x_store( feature_index ) + mu_x_store( feature_index );
		y_plot = y_vals * sigma_y + mu_y;

		% Plot
		figure;
		plot( x_plot, y_plot, 'LineWidth', 2 );
		xlabel( [ feature_names{ feature_index }, ' (original scale)' ] );
		ylabel( 'Predicted y (original scale)' );
		title( [ 'Partial Dependence (Custom GAM): ', feature_names{ feature_index } ] );
		grid on;
	
    end
	
	
	%% Implement the cluster-based mixture-of-GAMs model using the original feature
    
    L_values = 3:8;
    RMSE_per_L = zeros( length( L_values ), 1 );

    for idx_L = 1 : 1 : length( L_values )
	    num_clusters = L_values( 1, idx_L );
	    options = statset( 'Display', 'final', 'MaxIter', 1000, 'TolFun', 1e-6 );
    
	    % Step 1: Fit GMM on original x_data_train
	    gm_x_ori = fitgmdist( x_data_train, num_clusters, ...
		    'CovarianceType', 'diagonal', ...
		    'RegularizationValue', 1e-3, ...
		    'Replicates', 20, ...
		    'SharedCovariance', false, ...
		    'Start', 'plus', ...
		    'Options', options );
    
	    % Step 2: Get soft cluster responsibilities
	    Gamma_train = posterior( gm_x_ori, x_data_train );  % J_train × L
	    Gamma_test  = posterior( gm_x_ori, x_data_test );   % J_test × L
    
	    % Step 3: Train one GAM per cluster
	    gam_models_list = cell( num_clusters, 1 );
	    alpha_vals = zeros( 1, num_clusters );
    
	    lambda_gam = 1e0;
	    max_iter = 300;
	    knots_per_dim = 30;
	    rel_tol = 1e-3;
	    degree_gam = 3;
    
	    for ell = 1 : 1 : num_clusters
		    % Hard cluster assignments (could be refined using weighted regression later)
		    idx_ell = cluster( gm_x_ori, x_data_train ) == ell;
		    
		    X_ell = x_data_train( idx_ell, : );
		    y_ell = y_data_train( idx_ell );
    
		    [ alpha_ell, f_list_ell, delta_ell ] = fit_gam_backfitting( X_ell, y_ell, knots_per_dim, lambda_gam, max_iter, degree_gam, rel_tol );
		    
		    gam_models_list{ell} = f_list_ell;
		    alpha_vals( 1, ell ) = alpha_ell;
	    end
    
	    % Step 4: Predict test data from each GAM
	    Y_preds_std_test = zeros( J_test, num_clusters );
    
	    for ell = 1 : 1 : num_clusters
		    total_pred_test = zeros( J_test, 1 );
		    f_list_ell = gam_models_list{ell};
		    
		    for j = 1 : 1 : d
			    Phi_j = build_bspline_basis( x_data_test( :, j ), f_list_ell{j}.full_knots, degree_gam );
			    total_pred_test = total_pred_test + Phi_j * f_list_ell{j}.theta;
		    end
		    
		    Y_preds_std_test( :, ell ) = alpha_vals( 1, ell ) + total_pred_test;
	    end
    
	    % Step 5: Combine predictions via GMM responsibilities
	    y_hat_std_test = sum( Gamma_test .* Y_preds_std_test, 2 );  % Weighted sum
	    y_hat_test = y_hat_std_test * sigma_y + mu_y;  % Unstandardize
    
	    % Step 6: Evaluate RMSE
	    RMSE_test_x_ori = sqrt( mean( ( y_hat_test - y_data_test_ori ).^2 ) );
	    fprintf( 'Test RMSE (direct GMM-in-x + MGAM): %.5f\n', RMSE_test_x_ori );
        RMSE_per_L( idx_L, 1 ) = RMSE_test_x_ori;
    end
    RMSE_per_L
    
	
	