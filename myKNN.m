%function return error percentage for the given test data and k value
function [error] = myKNN(trainMat,testMat,k)
    dimensions = size(trainMat,2)-1; % number of dimensions present in data
    %training and testing set sizes
    trainSize = size(trainMat,1);
    testSize = size(testMat,1);
    %count number of classes in the input data
    numClasses=max(trainMat(:,dimensions+1))+1;
    
    %store k nearest euclidean distances and class labels in dist and labels 
    dist = -1*ones(testSize,k);
    labels = -1*ones(testSize,k);
    error=0;
    for i=1:testSize
        for j=1:trainSize
            %compute euclidean distance between ith test data point and jth train data point
            d = sqrt(sum((trainMat(j,1:dimensions) - testMat(i,1:dimensions)).^2));
            [dist(i,:),labels(i,:)] = insert(dist(i,:),labels(i,:),d,trainMat(j,dimensions+1),k);
        end
        %calculating the argmax for each labels(i,:)
        tempLabels = zeros(1,numClasses);
        for m=1:k
            tempLabels(labels(i,m)+1) = tempLabels(labels(i,m)+1)+1; 
        end
        [M,I]=max(tempLabels);
        %I is the class label for ith test data point
        if I~=testMat(i,dimensions+1)+1
            error = error+1;
        end
    end
    error = error/testSize*100;
end

%function which inserts the current training set data point in dist(i,:) at its
%appropriate position based on the distance 'd'
function [dist,labels]= insert(dist,labels,d,label,k)
    for i=1:k
        if dist(i)~=-1 && dist(i)>d
            if i<k
                dist(i+1:k)=dist(i:k-1);
                labels(i+1:k)=labels(i:k-1);
            end
            dist(i)=d;
            labels(i)=label;
            break;
        elseif dist(i)==-1
            dist(i)=d;
            labels(i)=label;
            break;
        end
    end
end