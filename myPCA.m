%The function returns the projection matrix 'W' and eigen values matrix 'eigVal'
%if 'numComp' is non zero then we return 'numComp' eigen vectors else we return
%all the eigen vectors
function [W,eigVal] = myPCA(dataMat,numComp)
    dimensions = size(dataMat,2)-1; % number of dimensions present in data
    dataSize = size(dataMat,1);

    %%calculating the mean of data
    mu = zeros(1,dimensions);
    for i=1:dataSize
        mu = mu + dataMat(i,1:dimensions);
    end
    mu = mu/dataSize;

    %%calculating the covariance of data
    covData = zeros(dimensions,dimensions);
    for i=1:dataSize
        covData = covData+((dataMat(i,1:dimensions)-mu)'*(dataMat(i,1:dimensions)-mu));
    end
    covData = covData/dataSize;

    %Calculate eigen vectors(V) and eigen values(D)
    [V,D]=eig(covData);
    %Sort the eigen values in descending order, sort the corresponding
    %eigen vectors as well
    for i=1:dimensions
        for j=i+1:dimensions
            if D(i,i)<D(j,j)
                tempD = D(i,i);
                D(i,i) = D(j,j);
                D(j,j) = tempD;

                tempV = V(:,i);
                V(:,i) = V(:,j);
                V(:,j) = tempV;
            end
        end
    end
    %If 'numComp' is non zero then we calculate numComp from minimum number of
    %eigen vectors that explain atleast 90% of the variance
    if numComp==0
        sumD = sum(sum(D));
        numerator = 0;
        X = zeros(1,dimensions);
        Y = zeros(1,dimensions);
        for i=1:dimensions
            numerator = numerator + D(i,i);
            X(i) = i;
            Y(i) = numerator/sumD;
        end
        plot(X,Y);
        title('2-D Plot');
        xlabel('Eigenvectors');
        ylabel('Proportion of variance');
        for i=1:dimensions
            if Y(i)>=0.9
                numComp=i;
                break;
            end
        end
        sprintf('Minimum number of principal components=%d',numComp)
    end
    %return the requested number of eigen vectors and eigen values
    W = V(:,1:numComp);
    eigVal = D(1:numComp,1:numComp);
end