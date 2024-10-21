pickFrom=[1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 10, 11, 11, 12];

t = 10000; % 10^4 samples of Zn

n = 100; % change value of n accordingly

samples = zeros(t, 1);

for i=1:t
    Xi=pickFrom(randi([1, 17], 1, n));
    Zn = sum(Xi) / n;
    % this is one sample of Zn
    samples(i) = Zn;
end

mu = 106/17;
variance = (3350/289)/n;
sigma = variance^0.5;
xi = linspace(0, 12, t); % Range of x values
pdf = normpdf(xi, mu, sigma);

histogram(samples, 'Normalization', 'pdf', 'BinWidth', (1/(n + 1)));
hold on;
plot(xi, pdf, 'Linewidth', 2);
title(['PDF of Zn for n = ', num2str(n)], ' with the PDF of a Gaussian RV');
xlabel("Value");
ylabel("Probability Density");
%legend("Histogram of Zn", "PDF of a Gaussian RV");
hold off;