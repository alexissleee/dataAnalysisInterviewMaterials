t = 10000; % 10^4 samples of Zn

n = 100; % change value of n accordingly

samples = zeros(t, 1);

for i=1:t
    Xn=rand(n, 1);
    Xn = 10 + Xn * 6; % to make it between 10 and 16
    Zn = sum(Xn) / n;
    % this is one sample of Zn
    samples(i) = Zn;
end

mu = 13;
variance = 3/n;
sigma = variance^0.5;
xi = linspace(10, 16, t); % Range of x values
pdf = normpdf(xi, mu, sigma);

histogram(samples, 'Normalization', 'pdf');
hold on;
plot(xi, pdf, 'Linewidth', 2);
title(['PDF of Zn for n = ', num2str(n)], ' with the PDF of a Gaussian RV');
xlabel("Value");
ylabel("Probability Density");
%legend("Histogram of Zn", "PDF of a Gaussian RV");
hold off;