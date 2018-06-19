function [z, w, v, trainErrorRate, validationErrorRate] = mlptrain(trainFile, validFile, m, k)
    %Load the training and validation data into matrices
    trainMat = dlmread(trainFile,',',0,0);
    validMat = dlmread(validFile,',',0,0);
    d = size(trainMat,2)-1; % number of dimensions present in data
    trainSize = size(trainMat,1);
    validSize = size(validMat,1);
    %Initial value of step size
    eta = 10^-4;
    %Randomly initialize weights between -0.01 and 0.01
    w = -0.01+(0.02)*rand(d+1,m);
    v = -0.01+(0.02)*rand(m+1,k);
    dw = zeros(size(w));
    dv = zeros(size(v));
    
    xt = zeros(d+1,1);
    %Induce bias in input and hidden units
    xt(1) = 1;
    z = zeros(trainSize,m+1);
    z(:,1)=1;
    y = zeros(k,1);
    E1=0;
    epoch=0;
    while 1
        epoch=epoch+1;
        E2=0;
        for t=randperm(trainSize)
            xt(2:d+1,1) = trainMat(t,1:d);
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
            
            %Back propogation and weights updation
            r = zeros(k,1);
            r(trainMat(t,d+1)+1)=1;
            tempSum(1,k) =0;
            for i=1:k
                dv(:,i) = eta*(r(i)-y(i))*z(t,:);
                tempSum(1,i) = (r(i)-y(i));
            end
            for h=2:m+1
                if w(:,h-1)'*xt<0
                    dw(:,h-1) = 0;
                else
                    dw(:,h-1) = eta*tempSum*v(h,:)'*xt;
                end
            end
            v = v+dv;
            w = w+dw;
            %Compute error for all the training data points
            E2 = E2 - r'*log(y);
        end
        %Loss function difference between current and previous epochs
        dE = abs(E2-E1);
        %For every 100 epochs decrease step size
        if mod(epoch,100)==0
            eta=eta/5;
        end
        %Convergence criteria
        if dE<10^-3
            break;
        end
        E1=E2;
    end
    %Feed forward for training set
    zz=zeros(1,m+1);
    zz(1,1)=1;
    errorRate=0;
    for t=1:trainSize
        xt(2:d+1,1) = trainMat(t,1:d);
        %Compute hidden unit values
        for h=2:m+1
            temp = w(:,h-1)'*xt;
            if temp<0
                zz(1,h)=0;
            else
                zz(1,h)=temp;
            end
        end
        %Compute output unit values
        y = v'*zz(1,:)';
        %Compute softmax
        denom = sum(exp(y));
        y = exp(y)/denom;
        [mx,ind] = max(y);
        %Compute misclassification error
        if ind-1~=trainMat(t,d+1)
            errorRate=errorRate+1;
        end
    end
    trainErrorRate=errorRate/trainSize*100;
    sprintf('Error rate on training set for m=%d is %d',m,trainErrorRate)
    
    %Feed forward for validation set
    zz=zeros(1,m+1);
    zz(1,1)=1;
    errorRate=0;
    for t=1:validSize
        xt(2:d+1,1) = validMat(t,1:d);
        %Compute hidden unit values
        for h=2:m+1
            temp = w(:,h-1)'*xt;
            if temp<0
                zz(1,h)=0;
            else
                zz(1,h)=temp;
            end
        end
        %Compute output unit values
        y = v'*zz(1,:)';
        %Compute softmax
        denom = sum(exp(y));
        y = exp(y)/denom;
        [mx,ind] = max(y);
        %Compute misclassification error
        if ind-1~=validMat(t,d+1)
            errorRate=errorRate+1;
        end
    end
    validationErrorRate=errorRate/validSize*100;
    sprintf('Error rate on validation set for m=%d is %d',m,validationErrorRate)
    w=w';
    v=v';
    z=z(:,2:m+1);
end