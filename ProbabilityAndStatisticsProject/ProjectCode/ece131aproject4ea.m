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

plot(xi, Zn, 'Linewidth', 2);
title(['Calculated PDF of Zn for n = ', num2str(n)]);
xlabel("Zn Value")
ylabel("Probability Density")