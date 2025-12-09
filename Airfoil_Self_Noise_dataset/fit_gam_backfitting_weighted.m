    
	function [ alpha, f_list, delta ] = fit_gam_backfitting_weighted( X, y, knots_per_dim, lambda, max_iter, degree, rel_tol, w_vec )
		[ N, d ] = size( X );
		f_list = cell( d, 1 );
		fj_vals = zeros( N, d );
        % degree = 3;
		w_vec = w_vec( : );
		w_sum = sum( w_vec );
		% alpha = mean( y );
		alpha = ( w_vec' * y ) / w_sum;

		for iter = 1 : 1 : max_iter
            fj_vals_old = fj_vals;  % Save previous estimate
			for j = 1 : 1 : d
				xj = X( :, j );
				lambda_j = lambda( 1, j );
                % qU = 0.0025;
                qU = 0;
                qL = 1 - qU;
				all_knots = quantile( xj, linspace( qU, qL, knots_per_dim + 2 ) );
				inner_knots = all_knots( 2 : end - 1 );
				% boundary_knots = [ min( xj ); max( xj ) ];
				boundary_knots = [ all_knots( 1 ); all_knots( end ) ];
                % If boundary knots collapse numerically, pad slightly
                if boundary_knots( 2, 1 ) - boundary_knots( 1, 1 ) < 1e-12
                    scale = max( 1, max( abs( [ boundary_knots( 1, 1 ), boundary_knots( 2, 1 ) ] ) ) );
                    delta = max( 1e-9, 1e-6 * scale );
                    boundary_knots = [ boundary_knots( 1, 1 ) - delta; boundary_knots( 2, 1 ) + delta ];
                end
				full_knots = [ repmat( boundary_knots( 1, 1 ), 1, degree + 1 ), inner_knots( : )', repmat( boundary_knots( 2, 1 ), 1, degree + 1 ) ];

				% interior_knots = unique( knots, 'stable' );
				Phi_j = build_bspline_basis( xj, full_knots, degree );
				% partial residual for coordinate j
				r_j = y - alpha - sum( fj_vals( :, [ 1 : j - 1, j + 1 : end ] ), 2 );
				
				% [ theta, f_j, ~ ] = fit_spline_ridge( Phi_j, r_j, lambda );
				% weighted ridge fit for this smooth
				[ theta, f_j, ~ ] = fit_spline_ridge_weighted( Phi_j, r_j, lambda_j, w_vec );
				
				% f_j = f_j - mean( f_j );  % Centering step
				c_j = ( w_vec' * f_j ) / w_sum;
				f_j = f_j - c_j;
				fj_vals( :, j ) = f_j;
				f_list{j} = struct( 'Phi', Phi_j, 'theta', theta, 'xj', xj, 'full_knots', full_knots, 'offset', c_j );
				
				% update intercept after all f_j are updated
				alpha = ( w_vec' * ( y - sum( fj_vals, 2 ) ) ) / w_sum;
			
			end
            % === Compute relative difference ===
            delta = norm( fj_vals(:) - fj_vals_old(:) ) / ( norm( fj_vals_old(:) ) + 1e-6 );
			if( ( delta < rel_tol ) && ( iter < max_iter ) )
				% fprintf( 'Iter %d, back-fitting converged with relative change = %.4e\n', iter, delta );
				break
			end
			if( ( delta >= rel_tol ) && ( iter >= max_iter ) )
				% fprintf( 'Iter %d, back-fitting did not converge with relative change = %.4e\n', iter, delta );
			end
		end
	end

	
	
	
	
	
	
	
	
	
	