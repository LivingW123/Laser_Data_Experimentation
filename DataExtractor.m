% Data Extractor | Version 1.2
% Todd Hao
close all;

% Change/enter your inputs here:
%-------------------------------------
% Gamma Value
gamma = 4

% Location of picture to analyze
picture = "C:\Users\mahar\Downloads\test_0026.tif"

% How far away from the curve will an outlier data point be replaced
maxOutlier = 3

% Location of the x200 matrix, used as import
x200Location = "C:\Users\mahar\Downloads\200x200.xlsx"

% Whether or not to rotate the image 180 degrees
rotate = true

% Where there is only one row of s1
oneS1 = true

% Top left of s1 coordinates (Down, Right) Leave as 0 to select it manually
topLeftx = 0
topLefty = 0

% Bottom right of s1 coordinates (Down, Right) Leave as 0 to select it manually
bottomRightx = 0
bottomRighty = 0

% Whether to manually select best fit curves
userSelect = false

% Whether to run 2 interpolations (Compared to 1 when false)
interpAgain = false

% The following should be these numbers most of the time
shrinkFactor = 22; 
finalLength = 48;
finalHeight = 24;

% Pixels above this value will be counted as saturated
maxBrightness = 230;

% Protected is the number of initial data points to not count as "bad"
% Useful as sometimes you do not want to change the first points
protected = 3
%------------------------------------

pic = imread(picture);

if rotate
    length = size(pic, 2);
    mid = floor(length/2);
    for (i=1:mid-1)
        temp = pic(:,i, :);
        pic(:,i, :) = pic(:,length+1-i, :);
        pic(:,length+1-i, :) = temp;
    end
end

fig = figure;
imshow(pic);

% If does we want to map coordinates manually
if ~(topLeftx||topLefty||bottomRightx||bottomRighty)

    datacursormode on
    dcm_obj = datacursormode(fig);
    
    set(dcm_obj,'UpdateFcn',@topFunction)
    
    disp('Click on the top left of the central two bands, then press "Return"')
    pause
    
    info_struct = getCursorInfo(dcm_obj);
    if isfield(info_struct, 'Position')
      disp('Clicked top left position is')
      disp(info_struct.Position)
      topLeft = info_struct.Position;
    end
    
    set(dcm_obj,'UpdateFcn',@bottomFunction)
    
    disp('Click on the bottom right of the central two bands, then press "Return"')
    disp("Note: Select a longer area than a shorter area")
    pause
    
    info_struct = getCursorInfo(dcm_obj);
    if isfield(info_struct, 'Position')
      disp('Clicked bottom right position is')
      disp(info_struct.Position)
      bottomRight = info_struct.Position;
    end
else
    topLeft = [topLeftx, topLefty];
    bottomRight = [bottomRightx, bottomRighty];
end

% expand the image up and down to height 528
growthFactor = floor((shrinkFactor*finalHeight-(bottomRight(2)-topLeft(2)))/2); % Amount of expansion up and down
picCrop = pic(topLeft(2)-growthFactor:bottomRight(2)+growthFactor, topLeft(1):bottomRight(1)); 

% Gamma transform
afterGamma = gammaTransform(picCrop, gamma);

% Shrink
shrunk = imresize(afterGamma, 1/shrinkFactor);

% If shrunk height is larger than needed, truncate it
[h,l]=size(shrunk);

if h-finalHeight>0
    shrunk=shrunk(1:finalHeight,:);
end

% If shrunk length is larger than needed, truncate it
if l-finalLength>0
    shrunk = shrunk(:,1:finalLength);
end

fig = figure;
imshow(shrunk);

% Calculate rows
shrunk = uint16(shrunk);

s1=(shrunk(floor(finalHeight/2), :) + shrunk(floor(finalHeight/2)+1,:))./2;

toAdd = 2;
if(oneS1)
    s1 = shrunk(floor(finalHeight/2),:);
    toAdd = 1;
end

s2=(shrunk(floor(finalHeight/2)-1, :) + shrunk(floor(finalHeight/2)+toAdd,:))./2;
s3=(shrunk(floor(finalHeight/2)-2, :) + shrunk(floor(finalHeight/2)+toAdd+1,:))./2;
s4=(shrunk(floor(finalHeight/2)-3, :) + shrunk(floor(finalHeight/2)+toAdd+2,:))./2;
s5=(shrunk(floor(finalHeight/2)-4, 1:8) + shrunk(floor(finalHeight/2)+toAdd+3,1:8))./2;

% s6 & s7 (bk)
s6=(shrunk(floor(finalHeight/2)-5, :) + shrunk(floor(finalHeight/2)+toAdd+4,:))./2;
s7=(shrunk(floor(finalHeight/2)-6, :) + shrunk(floor(finalHeight/2)+toAdd+5,:))./2;

% Remove Saturation
[s1SatX, s1] = removeSaturation(s1, maxBrightness);
[s2SatX, s2] = removeSaturation(s2, maxBrightness);
[s3SatX, s3] = removeSaturation(s3, maxBrightness);
[s4SatX, s4] = removeSaturation(s4, maxBrightness);
[s5SatX, s5] = removeSaturation(s5, maxBrightness);

% Subtract bk
s1bk = s1-s7(s1SatX);
s2bk = s2-s7(s2SatX);
s3bk = s3-s7(s3SatX);
s4bk = s4-s7(s4SatX);
s5bk = s5-s7(s5SatX);

% Find best fit
[s1Fit, s1Func, s1Error, s1mse] = findBestFit2(s1SatX, s1bk, finalLength, userSelect, "s1 Candidates");
[s2Fit, s2Func, s2Error, s2mse] = findBestFit2(s2SatX, s2bk, finalLength, userSelect, "s2 Candidates");
[s3Fit, s3Func, s3Error, s3mse] = findBestFit2(s3SatX, s3bk, finalLength, userSelect, "s3 Candidates");
[s4Fit, s4Func, s4Error, s4mse] = findBestFit2(s4SatX, s4bk, finalLength, userSelect, "s4 Candidates");
[s5Fit, s5Func, s5Error, s5mse] = findBestFit2(s5SatX, s5bk, 8, userSelect, "s5 Candidates");

%{
% Plot
figure();
plot(s1SatX, s1bk, 'bo', 'LineWidth', 1.5);
%{
hold on;
plot(s1Fit, 'r--', 'LineWidth', 1);
hold off;
%}
title("s1, Error: "+s1mse);
grid on;
xlabel("Pixels");
ylabel("Intensity");
legend("s1 Data", "s1 Best Fit: "+s1Func);

figure();
plot(s2SatX, s2bk, 'bo', 'LineWidth', 1.5);
%{
hold on;
plot(s2Fit, 'r--', 'LineWidth', 1);
hold off;
%}
title("s2, Error: "+s2mse);
grid on;
xlabel("Pixels");
ylabel("Intensity");
legend("s2 Data", "s2 Best Fit: "+s2Func);

figure();
plot(s3SatX, s3bk, 'bo', 'LineWidth', 1.5);
%{
hold on;
plot(s3Fit, 'r--', 'LineWidth', 1);
hold off;
%}
title("s3, Error: "+s3mse);
grid on;
xlabel("Pixels");
ylabel("Intensity");
legend("s3 Data", "s3 Best Fit: "+s3Func);

figure();
plot(s4SatX, s4bk, 'bo', 'LineWidth', 1.5);
%{
hold on;
plot(s4Fit, 'r--', 'LineWidth', 1);
hold off;
%}
title("s4, Error: "+s4mse);
grid on;
xlabel("Pixels");
ylabel("Intensity");
legend("s4 Data", "s4 Best Fit: "+s4Func);

figure();
plot(s5SatX, s5bk, 'bo', 'LineWidth', 1.5);
%{
hold on;
plot(s5Fit, 'r--', 'LineWidth', 1);
hold off;
%}
title("s5, Error: "+s5mse);
grid on;
xlabel("Pixels");
ylabel("Intensity");
legend("s5 Data", "s5 Best Fit: "+s5Func);
%}

% Interpolation 1

% Remove Outliers
s1Clean = removeOutliers(s1Error, s1bk, maxOutlier, protected);
s2Clean = removeOutliers(s2Error, s2bk, maxOutlier, protected);
s3Clean = removeOutliers(s3Error, s3bk, maxOutlier, protected);
s4Clean = removeOutliers(s4Error, s4bk, maxOutlier, protected);
s5Clean = removeOutliers(s5Error, s5bk, maxOutlier, protected);

% Condense arrays
[s1Condensed, s1x] = condenseArray(s1Clean);
[s2Condensed, s2x] = condenseArray(s2Clean);
[s3Condensed, s3x] = condenseArray(s3Clean);
[s4Condensed, s4x] = condenseArray(s4Clean);
[s5Condensed, s5x] = condenseArray(s5Clean);
%{
% Find the condensed best fit
[s1Fit2, s1Func, s1Error, s1mse] = findBestFit2(s1x, s1Condensed, finalLength, userSelect, "s1final Candidates");
[s2Fit2, s2Func, s2Error, s2mse] = findBestFit2(s2x, s2Condensed, finalLength, userSelect, "s2final Candidates");
[s3Fit2, s3Func, s3Error, s3mse] = findBestFit2(s3x, s3Condensed, finalLength, userSelect, "s3final Candidates");
[s4Fit2, s4Func, s4Error, s4mse] = findBestFit2(s4x, s4Condensed, finalLength, userSelect, "s4final Candidates");
[s5Fit2, s5Func, s5Error, s5mse] = findBestFit2(s5x, s5Condensed, 8, userSelect, "s5final Candidates");
%}
% Compile to get final
s1final1 = compile(s1x, s1Condensed, s1Fit);
s2final1 = compile(s2x, s2Condensed, s2Fit);
s3final1 = compile(s3x, s3Condensed, s3Fit);
s4final1 = compile(s4x, s4Condensed, s4Fit);
s5final1 = compile(s5x, s5Condensed, s5Fit);

%{
% Interpolation 2
if interpAgain
    
    % Remove Outliers
    s1Clean = removeOutliers(s1Error, s1final1, maxOutlier, protected);
    s2Clean = removeOutliers(s2Error, s2final1, maxOutlier, protected);
    s3Clean = removeOutliers(s3Error, s3final1, maxOutlier, protected);
    s4Clean = removeOutliers(s4Error, s4final1, maxOutlier, protected);
    s5Clean = removeOutliers(s5Error, s5final1, maxOutlier, protected);
    
    % Condense arrays
    [s1Condensed, s1x] = condenseArray(s1Clean);
    [s2Condensed, s2x] = condenseArray(s2Clean);
    [s3Condensed, s3x] = condenseArray(s3Clean);
    [s4Condensed, s4x] = condenseArray(s4Clean);
    [s5Condensed, s5x] = condenseArray(s5Clean);
    
    % Find the condensed best fit
    [s1Fit3, s1Func, s1Error, s1mse] = findBestFit2(s1x, s1Condensed, finalLength, userSelect, "s1final Candidates");
    [s2Fit3, s2Func, s2Error, s2mse] = findBestFit2(s2x, s2Condensed, finalLength, userSelect, "s2final Candidates");
    [s3Fit3, s3Func, s3Error, s3mse] = findBestFit2(s3x, s3Condensed, finalLength, userSelect, "s3final Candidates");
    [s4Fit3, s4Func, s4Error, s4mse] = findBestFit2(s4x, s4Condensed, finalLength, userSelect, "s4final Candidates");
    [s5Fit3, s5Func, s5Error, s5mse] = findBestFit2(s5x, s5Condensed, 8, userSelect, "s5final Candidates");
    
    % Compile to get final
    s1final = compile(s1x, s1Condensed, s1Fit3);
    s2final = compile(s2x, s2Condensed, s2Fit3);
    s3final = compile(s3x, s3Condensed, s3Fit3);
    s4final = compile(s4x, s4Condensed, s4Fit3);
    s5final = compile(s5x, s5Condensed, s5Fit3);
    
    % If any datapoint becomes negative, the second interpolation should be
    % ignored
    for i=1:48
        if (s1final(i)<0)
            s1final = s1final1;
            s1Fit3 = s1Fit2;
            s1mse = s1mse1;
            s1Func = s1Func1;
        end
        if (s2final(i)<0)
            s2final = s2final1;
            s2Fit3 = s2Fit2;
            s2mse = s2mse1;
            s2Func = s2Func1;
        end
        if (s3final(i)<0)
            s3final = s3final1;
            s3Fit3 = s3Fit2;
            s3mse = s3mse1;
            s3Func = s3Func1;
        end
        if (s4final(i)<0)
            s4final = s4final1;
            s4Fit3 = s4Fit2;
            s4mse = s4mse1;
            s4Func = s4Func1;
        end
    end
    
    for i=1:8
        if (s5final(i)<0)
            s5final = s5final1;
            s5Fit3 = s5Fit2;
            s5mse = s5mse1;
            s5Func = s5Func1;
            break;
        end
    end
    
    % If any data point becomes negative, the first interpolation should be
    % ignored
    for i=1:48
        if (s1final(i)<0)
            s1final = double(s1bk);
            s1Fit3 = s1Fit;
            s1mse = "Same as Original";
            s1Func = "Same as Original";
        end
        if (s2final(i)<0)
            s2final = double(s2bk);
            s2Fit3 = s2Fit;
            s2mse = "Same as Original";
            s2Func = "Same as Original";
        end
        if (s3final(i)<0)
            s3final = double(s3bk);
            s3Fit3 = s3Fit;
            s3mse = "Same as Original";
            s3Func = "Same as Original";
        end
        if (s4final(i)<0)
            s4final = double(s4bk);
            s4Fit3 = s4Fit;
            s4mse = "Same as Original";
            s4Func = "Same as Original";
        end
    end
    
    for i=1:8
        if (s5final(i)<0)
            s5final = double(s5bk);
            s5Fit3 = s5Fit;
            s5mse = "Same as Original";
            s5Func = "Same as Original";
            break;
        end
    end
else
    % Assign variables to graph
    s1final = s1final1;
    s2final = s2final1;
    s3final = s3final1;
    s4final = s4final1;
    s5final = s5final1;

    s1Fit3 = s1Fit2;
    s2Fit3 = s2Fit2;
    s3Fit3 = s3Fit2;
    s4Fit3 = s4Fit2;
    s5Fit3 = s5Fit2;

    s1mse = s1mse1;
    s2mse = s2mse1;
    s3mse = s3mse1;
    s4mse = s4mse1;
    s5mse = s5mse1;

    s1Func = s1Func1;
    s2Func = s2Func1;
    s3Func = s3Func1;
    s4Func = s4Func1;
    s5Func = s5Func1;

    % If any data point becomes negative, the first interpolation should be
    % ignored
    for i=1:48
        if (s1final1(i)<0)
            s1final = double(s1bk);
            s1Fit3 = s1Fit;
            s1mse = "Same as Original";
            s1Func = "Same as Original";
        end
        if (s2final1(i)<0)
            s2final = double(s2bk);
            s2Fit3 = s2Fit;
            s2mse = "Same as Original";
            s2Func = "Same as Original";
        end
        if (s3final1(i)<0)
            s3final = double(s3bk);
            s3Fit3 = s3Fit;
            s3mse = "Same as Original";
            s3Func = "Same as Original";
        end
        if (s4final1(i)<0)
            s4final = double(s4bk);
            s4Fit3 = s4Fit;
            s4mse = "Same as Original";
            s4Func = "Same as Original";
        end
    end
    
    for i=1:8
        if (s5final1(i)<0)
            s5final = double(s5bk);
            s5Fit3 = s5Fit;
            s5mse = "Same as Original";
            s5Func = "Same as Original";
            break;
        end
    end
end
%}

% Plot Again
figure();
plot(s1final1, 'bo', 'LineWidth', 1.5);
hold on;
plot(s1Fit, 'r--', 'LineWidth', 1);
hold off;
title("s1Final, Error: "+s1mse);
grid on;
xlabel("Pixels");
ylabel("Intensity");
legend("s1 Data", "s1 Best Fit: "+s1Func);
ylim([0 35]);

figure();
plot(s2final1, 'bo', 'LineWidth', 1.5);
hold on;
plot(s2Fit, 'r--', 'LineWidth', 1);
hold off;
title("s2Final, Error: "+s2mse);
grid on;
xlabel("Pixels");
ylabel("Intensity");
legend("s2 Data", "s2 Best Fit: "+s2Func);

figure();
plot(s3final1, 'bo', 'LineWidth', 1.5);
hold on;
plot(s3Fit, 'r--', 'LineWidth', 1);
hold off;
title("s3Final, Error: "+s3mse);
grid on;
xlabel("Pixels");
ylabel("Intensity");
legend("s3 Data", "s3 Best Fit: "+s3Func);

figure();
plot(s4final1, 'bo', 'LineWidth', 1.5);
hold on;
plot(s4Fit, 'r--', 'LineWidth', 1);
hold off;
title("s4Final, Error: "+s4mse);
grid on;
xlabel("Pixels");
ylabel("Intensity");
legend("s4 Data", "s4 Best Fit: "+s4Func);

figure();
plot(s5final1, 'bo', 'LineWidth', 1.5);
hold on;
plot(s5Fit, 'r--', 'LineWidth', 1);
hold off;
title("s5Final, Error: "+s5mse);
grid on;
xlabel("Pixels");
ylabel("Intensity");
legend("s5 Data", "s5 Best Fit: "+s5Func);

% Prepare data for TSVD
s1final = s1final1';
s2final = s2final1';
s3final = s3final1';
s4final = s4final1';
s5final = s5final1';

sample = readtable(x200Location);
x200 = table2array(sample);

function output_txt = topFunction(~,event_obj)
  % ~            Currently not used (empty)
  % event_obj    Object containing event data structure
  % output_txt   Data cursor text
  pos = get(event_obj, 'Position');
  output_txt = {["Top left is: "], ['x: ' num2str(pos(1))], ['y: ' num2str(pos(2))]};
end

function output_txt = bottomFunction(~,event_obj)
  % ~            Currently not used (empty)
  % event_obj    Object containing event data structure
  % output_txt   Data cursor text
  pos = get(event_obj, 'Position');
  output_txt = {["Bottom right is: "], ['x: ' num2str(pos(1))], ['y: ' num2str(pos(2))]};
end

function afterGamma = gammaTransform(beforeGamma, gamma)
    % Gamma transforms the image, beforeGamma, by the value of gamma

    picDouble = im2double(beforeGamma);
    afterGamma1 = abs((1*picDouble).^gamma);
    maxr = max(afterGamma1(:));
    minc = min(afterGamma1(:));
    afterGamma = uint8((255.*afterGamma1)./(maxr-minc));
end

function [fit, func, error]=findBestFit1(data, numDataPoints, userSelect, graphTitle)
    % Finds the curve of best fit for data, with numDataPoints data
    % Inputs: userSelect, Boolean of whether to manually select
    %         title, the title indicating the data of each graph, s1-s5
    % Outputs: fit, the fit datapoints as an array of length numDataPoints
    %          func, a string representing the function chosen
    %          error, an array of length numDataPoints of errors of each data point
    data = double(data);

    poly5Para=polyfit(1:numDataPoints, data, 5);
    poly5Fit = polyval(poly5Para, 1:numDataPoints);
    poly5Mse = mean((data-poly5Fit).^2);
    poly5Error = data-poly5Fit;

    poly6Para=polyfit(1:numDataPoints, data, 6);
    poly6Fit = polyval(poly6Para, 1:numDataPoints);
    poly6Mse = mean((data-poly6Fit).^2);
    poly6Error = data-poly6Fit;

    poly7Para=polyfit(1:numDataPoints, data, 7);
    poly7Fit = polyval(poly7Para, 1:numDataPoints);
    poly7Mse = mean((data-poly7Fit).^2);
    poly7Error = data-poly7Fit;

    gammaFitPara=lsqcurvefit(@gammaVariate, [0 1 1 1], 1:numDataPoints, data);
    gammaFit = gammaVariate(gammaFitPara, 1:numDataPoints);
    gammaMse = mean((data-gammaFit).^2);
    gammaError = data-gammaFit;

    kevinFitPara=lsqcurvefit(@kevinFunc, [1 1 0], 1:numDataPoints, data);
    kevinFit = kevinFunc(kevinFitPara, 1:numDataPoints);
    kevinMse = mean((data-kevinFit).^2);
    kevinError = data-kevinFit;

    if userSelect
        figure;
        plot(1:numDataPoints, data, 'ro');
        hold on;
        plot(1:numDataPoints, poly5Fit);
        plot(1:numDataPoints, poly6Fit);
        plot(1:numDataPoints, poly7Fit);
        plot(1:numDataPoints, gammaFit);
        plot(1:numDataPoints, kevinFit);
    
        legend("Data", "poly5", "poly6", "poly7", "gamma", "kevin")
        title(graphTitle)
        hold off;
    
        disp("1 --> poly5")
        disp("2 --> poly6")
        disp("3 --> poly7")
        disp("4 --> gamma")
        disp("5 --> kevin")

        num = input("Please enter a number: ");
        switch(num)
            case 1
                fit = poly5Fit;
                func = "poly5";
                error = poly5Error;
            case 2
                fit = poly6Fit;
                func = "poly6";
                error = poly6Error;
            case 3
                fit = poly7Fit;
                func = "poly7";
                error = poly7Error;
            case 4
                fit = gammaFit;
                func = "gamma";
                error = gammaError;
            case 5
                fit = kevinFit;
                func = "kevin";
                error = kevinError;
        end
        close;

    else
        % Don't use poly7 for less than 8 data points
        if (numDataPoints > 8)
            bestError = min([poly5Mse, poly6Mse, poly7Mse, gammaMse, kevinMse]);
        else
            bestError = min([poly5Mse, poly6Mse, gammaMse, kevinMse]);
        end
        
        % Adjusting for if bestError is nonreal
        if (~isreal(bestError))
            if (numDataPoints > 8)
                bestError = min([poly5Mse, poly6Mse, poly7Mse, kevinMse]);
            else
                bestError = min([poly5Mse, poly6Mse, kevinMse]);
            end
        end
        
        % Output based on smallest error
        switch(bestError)
            case poly5Mse
                fit = poly5Fit;
                func = "poly5";
                error = poly5Error;
            case poly6Mse
                fit = poly6Fit;
                func = "poly6";
                error = poly6Error;
            case poly7Mse
                fit = poly7Fit;
                func = "poly7";
                error = poly7Error;
            case gammaMse
                fit = gammaFit;
                func = "gamma";
                error = gammaError;
            case kevinMse
                fit = kevinFit;
                func = "kevin";
                error = kevinError;
        end
    end
end

function [fit, func, error, mse]=findBestFit2(x, data, fitLength, userSelect, graphTitle)
    % Finds the curve of best fit for incomplete data
    % Inputs: x, corresponding x coordinates of data
    %         data, corresponding values of x coordinates of data
    %         fitLength, the length to return
    % Outputs: fit, a complete array of the fit data of length fitLength
    %          func, a string representing the fit chosen
    %          error, a complete array of error corresponding to each data point of fit
    %          mse, the mean squared error of the fit, as a number
    
    data = double(data);

    poly5Para=polyfit(x, data, 5);
    poly5Fit = polyval(poly5Para, 1:fitLength);
    poly5ErrorArray = zeros(1, length(data));
    for i = 1:length(data)
        poly5ErrorArray(i) = abs(data(i)-poly5Fit(x(i)));
    end
    poly5Error = mean(poly5ErrorArray.^2);
    poly5ErrorCompleteA = compile(x, poly5ErrorArray, zeros([1,48]));

    poly6Para=polyfit(x, data, 6);
    poly6Fit = polyval(poly6Para, 1:fitLength);
    poly6ErrorArray = zeros(1, length(data));
    for i = 1:length(data)
        poly6ErrorArray(i) = abs(data(i)-poly6Fit(x(i)));
    end
    poly6Error = mean(poly6ErrorArray.^2);
    poly6ErrorCompleteA = compile(x, poly6ErrorArray, zeros([1,48]));

    poly7Para=polyfit(x, data, 7);
    poly7Fit = polyval(poly7Para, 1:fitLength);
    poly7ErrorArray = zeros(1, length(data));
    for i = 1:length(data)
        poly7ErrorArray(i) = abs(data(i)-poly7Fit(x(i)));
    end
    poly7Error = mean(poly7ErrorArray.^2);
    poly7ErrorCompleteA = compile(x, poly7ErrorArray, zeros([1,48]));

    gammaFitPara=lsqcurvefit(@gammaVariate, [0 1 1 1], x, data);
    gammaFit = gammaVariate(gammaFitPara, 1:fitLength);
    gammaErrorArray = zeros(1, length(data));
    for i = 1:length(data)
        gammaErrorArray(i) = abs(data(i)-gammaFit(x(i)));
    end
    gammaError = mean(gammaErrorArray.^2);
    gammaErrorCompleteA = compile(x, gammaErrorArray, zeros([1,48]));

    kevinFitPara=lsqcurvefit(@kevinFunc, [1 1 0], x, data);
    kevinFit = kevinFunc(kevinFitPara, 1:fitLength);
    kevinErrorArray = zeros(1, length(data));
    for i = 1:length(data)
        kevinErrorArray(i) = abs(data(i)-kevinFit(x(i)));
    end
    kevinError = mean(kevinErrorArray.^2);
    kevinErrorCompleteA = compile(x, kevinErrorArray, zeros([1,48]));

    if userSelect
        figure;
        plot(x, data, 'ro');
        hold on;
        plot(1:fitLength, poly5Fit);
        plot(1:fitLength, poly6Fit);
        plot(1:fitLength, poly7Fit);
        plot(1:fitLength, gammaFit)
        plot(1:fitLength, kevinFit);
        axis([0,fitLength, 0, max(data)+10])
        title(graphTitle)
        legend("Data", "poly5", "poly6", "poly7", "gamma", "kevin")
        hold off;

        disp("1 --> poly5")
        disp("2 --> poly6")
        disp("3 --> poly7")
        disp("4 --> gamma")
        disp("5 --> kevin")

        num = input("Please enter a number: ");
        switch(num)
            case 1
                fit = poly5Fit;
                func = "poly5";
                mse = poly5Error;
                error = poly5ErrorCompleteA;
            case 2
                fit = poly6Fit;
                func = "poly6";
                mse = poly6Error;
                error = poly6ErrorCompleteA;
            case 3
                fit = poly7Fit;
                func = "poly7";
                mse = poly7Error;
                error = poly7ErrorCompleteA;
            case 4
                fit = gammaFit;
                func = "gamma";
                mse = gammaError;
                error = gammaErrorCompleteA;
            case 5
                fit = kevinFit;
                func = "kevin";
                mse = kevinError;
                error = kevinErrorCompleteA;
        end
        close;
        
    else
        if (fitLength > 8)
            bestError = min([poly5Error, poly6Error, poly7Error, kevinError]);
        else
            bestError = min([poly5Error, poly6Error, kevinError]);
        end
    
        switch(bestError)
            case poly5Error
                fit = poly5Fit;
                func = "poly5";
                error = poly5ErrorCompleteA;
                mse = poly5Error;
            case poly6Error
                fit = poly6Fit;
                func = "poly6";
                error = poly6ErrorCompleteA;
                mse = poly6Error;
            case poly7Error
                fit = poly7Fit;
                func = "poly7";
                error = poly7ErrorCompleteA;
                mse = poly7Error;
            case kevinError
                fit = kevinFit;
                func = "kevin";
                error = kevinErrorCompleteA;
                mse = kevinError;
        end
    end
end

% y = b(x-a)^c*e^(a-x/d)
function y = gammaVariate(beta, x)
    y = beta(2).*((x-beta(1)).^beta(3)).*exp(-(x-beta(1))./beta(4));
end

% y=a*exp(-b*(x-c)*(x-c)/x)
function y = kevinFunc(beta, x)
    y = beta(1).*exp(-beta(2).*((x-beta(3)).^2)./x);
end

% Protected is the number of initial data points to not count as "bad"
function y = removeOutliers(error, data, maxOutlier, protected)
    % Removes outliers in an array of data and their corresponding errors
    % Inputs: error, the errors of each datapoint
    %         data, the data corresponding to each error value
    %         maxOutlier, the max outlier to check for removal
    %         protected, the number of first datapoints to not touch
    % Outputs: y, the array with outliers removed as NaNs
    y = NaN([1,length(data)]);
    for i=1:length(y)
        if abs(error(i))<=maxOutlier || i<=protected
            y(i) = data(i);
        end
    end
end

function final = compile(x, data, fit)
    % Compiles the incomplete data array and fills it in with fit data
    % Inputs: x, the corresponding x values of data array elements
    %         data, the real data we want to use
    %         fit, the substitute when data for some datapoint does not
    %         exist
    % Outputs: final, the compiled array with all elements from data and
    %          some from fit

    final = fit;
    for i = 1:length(x)
        final(x(i)) = data(i);
    end
end

function [condensed, x] = condenseArray(data)
    % Condenses data into a connected array of data
    % Inputs: data, an array containing NaNs
    % Outputs: Condensed, a condensed array, with NaNs removed
    %          x, the corresponding x values of each value in condensed
    
    x = 1:length(data);
    condensed = data;
    toDelete = [];
    for i = 1:length(data)
        if isnan(data(i))
            toDelete(end+1) = i;
        end
    end
    condensed(toDelete) = [];
    x(toDelete) = [];
end

function [x, s] = removeSaturation(data, maxSaturation)
    % Removes saturated pixels from data
    % Inputs: data, an array of (potentially) saturated data
    %         maxSaturation, the value above which pixels will be
    %         classified as saturated
    % Outputs: x, the array corresponding to nonsaturated pixels' coordinates
    %          s, the array with saturated pixels removed


    x = 1:length(data);
    x = x(data<=maxSaturation);
    s = data(x);
end