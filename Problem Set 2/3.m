N = 1000;
beta = 1.0;
gamma = 1/3;
dt = 0.1;
T_max = 60;
nSteps = round(T_max / dt);
t_Euler = linspace(0, T_max, nSteps+1);
S = zeros(1, nSteps+1);
I = zeros(1, nSteps+1);
R = zeros(1, nSteps+1);
S(1) = 995; I(1) = 5; R(1) = 0;

for k = 1:nSteps
    S_k = S(k); I_k = I(k); R_k = R(k);
    dS = -beta * S_k * I_k / N;
    dI = beta * S_k * I_k / N - gamma * I_k;
    dR = gamma * I_k;
    S(k+1) = S_k + dt * dS;
    I(k+1) = I_k + dt * dI;
    R(k+1) = R_k + dt * dR;
end

figure;
plot(t_Euler, S, 'b-', 'LineWidth', 1.5); hold on;
plot(t_Euler, I, 'r-', 'LineWidth', 1.5);
plot(t_Euler, R, 'g-', 'LineWidth', 1.5);
xlabel('Time (days)');
ylabel('Number of Individuals');
title('SIR Model (Forward Euler, \beta=1, \gamma=1/3)');
legend('S', 'I', 'R', 'Location', 'Best');
grid on;

U0 = [995; 5; 0];
tspan = [0, 60];
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t_ode, U_ode] = ode23(@sir_rhs, tspan, U0, opts);
S_ode = U_ode(:,1);
I_ode = U_ode(:,2);
R_ode = U_ode(:,3);

figure;
plot(t_ode, S_ode, 'b-', 'LineWidth', 1.5); hold on;
plot(t_ode, I_ode, 'r-', 'LineWidth', 1.5);
plot(t_ode, R_ode, 'g-', 'LineWidth', 1.5);
xlabel('Time (days)');
ylabel('Number of Individuals');
title('SIR Model (ode23 Solver, \beta=1, \gamma=1/3)');
legend('Susceptible', 'Infected', 'Recovered', 'Location', 'Best');
grid on;

function dUdt = sir_rhs(~, U)
    S_val = U(1);
    I_val = U(2);
    R_val = U(3);
    N_loc = 1000;
    beta_loc = 1.0;
    gamma_loc = 1/3;
    dS = -beta_loc * S_val * I_val / N_loc;
    dI = beta_loc * S_val * I_val / N_loc - gamma_loc * I_val;
    dR = gamma_loc * I_val;
    dUdt = [dS; dI; dR];
end