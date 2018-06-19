%The function returns the projection matrix 'W' and eigen values matrix 'eigVal'
%if 'numComp' is non zero then we return 'numComp' eigen vectors else we return
%all the eigen vectors
function [W,eigVal] = myLDA(dataMat,numComp)
    dimensions = size(dataMat,2)-1; % number of dimensions present in data
    dataSize = size(dataMat,1);
    %calculate mean and size of eah class present in data
    mu = zeros(10,dimensions);
    N = zeros(10,1);
    for i=1:dataSize
        mu(dataMat(i,dimensions+1)+1,:) = mu(dataMat(i,dimensions+1)+1,:) + dataMat(i,1:dimensions);
        N(dataMat(i,dimensions+1)+1,1) = N(dataMat(i,dimensions+1)+1,1) + 1;
    end
    for i=1:10
        mu(i,:) = mu(i,:)/N(i,1);
    end
    %calculate global mean
    m = sum(mu)/10;
    
    %Calculate within-class scatter
    Sw = zeros(dimensions,dimensions);
    for i=1:dataSize
        Sw = Sw + (dataMat(i,1:dimensions)-mu(dataMat(i,dimensions+1)+1,:))'*(dataMat(i,1:dimensions)-mu(dataMat(i,dimensions+1)+1,:));
    end
    
    %Calculate between class scatter
    Sb = zeros(dimensions,dimensions);
    for i=1:10
        Sb = Sb + N(i)*(mu(i,:)-m)'*(mu(i,:)-m);
    end
    
    %calculate eigen vectors. Sw is non invertible so calculate psuedo
    %inverse
    %'V' is eigen vectors and 'D' is eigen values
    [V,D]=eig(pinv(Sw)*Sb);
    
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
    %return the requested number of eigen vectors and eigen values
    W = V(:,1:numComp);
    eigVal = D(1:numComp,1:numComp);
end