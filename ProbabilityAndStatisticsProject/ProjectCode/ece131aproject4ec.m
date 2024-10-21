n = 100; % change value of n accordingly
t = 100;

Xi = ones(t, 1); 
Xi = Xi / 6; %pdfs of Xi's
Zn = Xi;

if n > 1
    for i=2:n
        Zn = conv(Zn, Xi);
    end
end

xi = linspace(10, 16, length(Zn));
areaUnderGraph = trapz(xi, Zn);
Zn = Zn / areaUnderGraph; % normalize Zn

mu = 13;
variance = 3/n;
sigma = variance^0.5;
pdf = normpdf(xi, mu, sigma);

plot(xi, Zn, 'Linewidth', 2);
hold on;
plot(xi, pdf, 'Linewidth', 2);
title(['Calculated PDF of Zn for n = ', num2str(n)], ' with the PDF of a Gaussian RV');
xlabel("Value");
ylabel("Probability Density");
hold off;