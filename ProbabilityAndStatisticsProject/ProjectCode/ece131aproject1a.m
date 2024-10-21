numTosses = 100000; % change numTosses value accordingly
X1=randi([1, 12], 1, numTosses);
histogram(X1);
title(['t = ', num2str(numTosses), ' tosses']);
xlabel('Outcome');
ylabel('# of times');