function [z] = mlptest(testFile,w,v)
    %Load the test data into matrix
    testMat = dlmread(testFile,',',0,0);
    w=w';
    v=v';
    m = size(w,2);
    d = size(testMat,2)-1; % number of dimensions present in data
    testSize = size(testMat,1);
    z=zeros(testSize,m+1);
    z(:,1)=1;
    errorRate=0;
    %feedforward for input data
    for t=1:testSize
        xt(2:d+1,1) = testMat(t,1:d);
        %Compute hidden unit values
        for h=2:m+1
            temp = w(:,h-1)'*xt;
            if temp<0
                z(t,h)=0;
            else
                z(t,h)=temp;
            end
        end
        %Compute output unit values
        y = v'*z(t,:)';
        %Compute softmax
        denom = sum(exp(y));
        y = exp(y)/denom;
        [mx,ind] = max(y);
        %misclassification Error
        if ind-1~=testMat(t,d+1)
            errorRate=errorRate+1;
        end
    end
    z=z(:,2:m+1);
    errorRate=errorRate/testSize*100;
    sprintf('Error rate on input data is %d',errorRate)
end