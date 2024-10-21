t = 10000; % 10^4 samples of Zn

n = 1; % change value of n accordingly

samples = zeros(t, 1);

for i=1:t
    Xn=rand(n, 1);
    Xn = 10 + Xn * 6; % to make it between 10 and 16
    Zn = sum(Xn) / n;
    % this is one sample of Zn
    samples(i) = Zn;
end

histogram(samples, 'Normalization', 'pdf');
title(['PDF of Zn for n = ', num2str(n)]);
xlabel("Zn Value")
ylabel("Probability Density")