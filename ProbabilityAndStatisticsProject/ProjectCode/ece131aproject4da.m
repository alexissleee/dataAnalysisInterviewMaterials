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

histogram(samples, 'Normalization', 'pdf', 'BinWidth', (1/(n + 1)));
title(['PDF of Zn for n = ', num2str(n)]);
xlabel("Zn Value")
ylabel("Probability Density")