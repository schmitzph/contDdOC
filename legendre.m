%%%% legendre.m
%%%%
%%%% This file contains a numerical example to demonstrate the data-driven
%%%% optimal control approach based on a continuous-time fundamental lemma
%%%%
%%%% by Philipp Schmitz
%%%%
%%%% requieres Optimization Toolbox
%%%% tested with MATLAB 23.2 (R2023b maca64)
%%%%
%%%% 2024-07-04

%%%% Optimal Control Problem (OCP) %%%%

%%%% dot x = -x + u
%%%% x(-1) = 1
%%%% min int_{-1}^{1} ||x||^2 + ||u||^2

close all
clear

% options for fmincon
options = optimoptions('fmincon','ConstraintTolerance',1e-15, 'OptimalityTolerance', 1e-15, 'Display','none');

tspan = -1:0.1:1;

%%% the optimal solution can by analyticaly calculated using
%%% Pontryargin's minimum principle
%%%
%%% dot lambda = lambda - x
%%% lambda(1) = 0
%%% u = -lambda

% exact optimal solution (uopt,xopt)
x_ = @(t) exp(-sqrt(2)*t) .* ( (sqrt(2)-2)*exp(2*sqrt(2)*t) - (sqrt(2)+2)*exp(2*sqrt(2)) ) / ( sqrt(2) * ( exp(2*sqrt(2)) - 1 ) );
x_opt = @(t) x_(t)/x_(-1); % normalization in order to have x(-1)=1

lambda_ = @(t) exp(-sqrt(2)*t).*( exp(2*sqrt(2)*t) - exp(2*sqrt(2)) ) / ( exp(2*sqrt(2)) - 1 );
lambda = @(t) lambda_(t)/x_(-1);

u_opt = @(t) -lambda(t);

% stage cost
ell = @(t) u_opt(t).^2 + x_opt(t).^2;

% optimal cost
J_opt = integral(ell,-1,1);




%%% we calculate on approximated optimal solution the OCP
%%% by means of Legendre polynomials
%%% employing the system model

% number of Legendre polynomials involved
N = 5;
fprintf('\npolynomial order:\t%i\n',N)

% truncated differential operator in coefficient space
D = diff_coef(N);


% formulate pollynomially restricted model-based optimal control problem
% as quadratic programm and solve with fmincon

% cost
Jfun = @(c) cost_coef(c(1:N),c(N+1:2*N));

% system model
Aeq = [-eye(N), eye(N)+D; zeros(1,N), (-1).^(0:N-1)];
beq = [zeros(N,1);1];

[c_ux,J_approx,exitflag]=fmincon(Jfun,rand(1,2*N),[],[],Aeq,beq,[],[],[],options);

% recover time-domain trajectory from series coefficients
u_approx = @(t) series_expan(t, c_ux(1:N));
x_approx = @(t) series_expan(t, c_ux(N+1:2*N));



%%% we calculate on approximated optimal solution the OCP
%%% by means of Legendre polynomials
%%% using a data-driven description instead of the model (fundamental lemma)

% data trajectory (u_dat, x_dat)
% % dx = -x + u
% x(-1) = 0

% input with higher order derivatives
u_dat = @(t) t.^2;
du_dat = @(t) 2*t;
ddu_dat = @(t) 2;
% u_dat is persistenly exciting of order L=3

% state with higher order derivatives
x_dat = @(t) t.^2-2*t-5*exp(-t-1)+2;
dx_dat = @(t) -x_dat(t) + u_dat(t);
ddx_dat = @(t) -dx_dat(t) + du_dat(t);

% calculate Gramian
Lambda_u = @(t) [u_dat(t); du_dat(t); ddu_dat(t)];
sqLambda_u = @(t) Lambda_u(t) .* Lambda_u(t)';
Gamm_u = integral(sqLambda_u, -1, 1, 'ArrayValued',true);

fprintf('smallest ev of Gamm_u:\t%f\n', min(eig(Gamm_u)))

% calculate combined Gramian serving as data-based model surrogate
Lambda_ux = @(t) [u_dat(t); x_dat(t); dx_dat(t)];
sqLambda_ux = @(t) Lambda_ux(t) .* Lambda_ux(t)';
Gamm_ux = integral(sqLambda_ux, -1, 1, 'ArrayValued',true);

fprintf('rank Gamm_ux:\t%i\n', rank(Gamm_ux))

% decompose Gamm_ux
Gu = Gamm_ux(1,:);
Gx = Gamm_ux(2,:);
Gdx = Gamm_ux(3,:);

% formulating the data-driven pollynomially restricted OCP
% as quadratic programm;
% the polynimial order N is the same as before

Aeq_dd = [D*kron(eye(N),Gx)-kron(eye(N),Gdx); (-1).^(0:N-1)*kron(eye(N),Gx)];
beq_dd = [zeros(N,1); 1];
J_dd_fun = @(g) cost_data(g, Gx, Gu, N);

% and solve it
[g,J_dd,exitflag]=fmincon(J_dd_fun,zeros(1,3*N),[],[],Aeq_dd,beq_dd,[],[],[],options);
cx_dd = kron(eye(N),Gx)*g';
cu_dd = kron(eye(N),Gu)*g';

% recover the trajectory from the coefficients
u_dd = @(t) series_expan(t, cu_dd);
x_dd = @(t) series_expan(t, cx_dd);



fig=figure;
subplot(2,1,1)
hold on
plot(tspan, u_dd(tspan), 'b', 'LineStyle','-', 'LineWidth', 1.5)
plot(tspan, u_approx(tspan), 'r', 'Marker','o', 'LineStyle', 'None', 'LineWidth', 1.5)
plot(tspan, u_opt(tspan), 'k', 'LineStyle','--', 'LineWidth', 1.5)
legend('data-driven', 'model-based', 'exact')
xlabel('t')
ylabel('input u')

subplot(2,1,2)
hold on
plot(tspan, x_dd(tspan), 'b', 'LineStyle','-', 'LineWidth', 1.5)
plot(tspan, x_approx(tspan), 'r', 'Marker','o', 'LineStyle', 'None', 'LineWidth', 1.5)
plot(tspan, x_opt(tspan), 'k', 'LineStyle','--', 'LineWidth', 1.5)
legend('data-driven', 'model-based', 'exact')
xlabel('t')
ylabel('state x')


fprintf('\noptimality gap\ndata_driven:\t%.3e\nmodel-based:\t%.3e\n\n',J_dd-J_opt,J_approx-J_opt)


function c = cost_data(g, Gx, Gu, N)
    m = size(Gx,2);
    c = sum((kron(diag(sqrt(sqnorm(0:N-1))), Gu) * g').^2);
    c = c + sum((kron(diag(sqrt(sqnorm(0:N-1))), Gx) * g').^2);
end

function c = coef(fun,n)
% calculate coefficients up to order n of series expansion for function fun
    c = zeros(1,n);
    for i=1:n
        g = @(t) legendreP(i-1,t) .* fun(t);
        c(i) = integral(g,-1,1)/sqnorm(i-1); % with normalization
    end
end

function y = series_expan(t,c)
% given expansion coefficients evaluate, truncated series at t
    y = zeros(1,length(t));
    n = length(c);
    for i=1:n
        y = y + legendreP(i-1,t)*c(i);
    end
end

function D = diff_coef(n)
% build matrix representing truncated differential operator in coefficient
% space of order n
    D = zeros(n);
    for j = 1:n
        D(j,j+1:2:end) = 2*j-1;
    end
end

function sn = sqnorm(n)
% sqared norm of the n-th Legendre polynomial
    sn = 2./(2*n+1);
end

function J = cost_coef(cu,cx)
% cost in terms of expansion coefficients
    nu = length(cu);
    nx = length(cx);
    Ju = sum(cu.^2.*sqnorm(0:nu-1));
    Jx = sum(cx.^2.*sqnorm(0:nx-1));
    J = Jx + Ju;
end