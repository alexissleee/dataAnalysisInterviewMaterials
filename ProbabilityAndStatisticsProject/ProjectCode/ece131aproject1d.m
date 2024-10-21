pickFrom=[1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 10, 11, 11, 12];

numTosses = 100000; % change value of numTosses accordingly
X1=pickFrom(randi([1, 17], 1, numTosses));

histogram(X1);
title(['t = ', num2str(numTosses), ' tosses']);
xlabel('Outcome');
ylabel('# of times');