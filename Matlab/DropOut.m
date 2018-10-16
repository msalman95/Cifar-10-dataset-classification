function y = DropOut(x,prob)
    check = rand(size(x));
    check(check < prob) = 0;
    check(check >= prob) = 1;
    y = x.*check/prob;
end

    
