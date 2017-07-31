% create a vector of N points between a and b
a=-1.2; b = -a; N = 100; X=linspace(a, b, N);

Y = X.^3 - X; % a function of X

figure(1); clf; % pop up a figure, clean it

H = axes; set(H, 'fontsize', 20); % set the font

plot(X, Y, 'color', 'blue', 'linewidth', 2); % plot

axis([a, b, -0.6, 0.6]); % the viewable box

saveas(gcf, 'MyPicture.eps', 'psc2'); % save the picture in color