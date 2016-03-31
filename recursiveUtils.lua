
function torch.recursiveapply(x,func,z)
    if type(x) == 'table' then
        local y = {}
        for k,v in pairs(x) do
            local zk
            if z then
                zk = z[k]
            end
            y[k] = torch.recursiveapply(v,func,zk)
        end
        return y
    else
        return func(x,z)
    end
end
function torch.recursiveaccumulate(x,func,z)
    if type(x) == 'table' then
        local y = 0
        for k,v in pairs(x) do
            local zk
            if z then
                zk = z[k]
            end
            y = torch.recursiveaccumulate(v,func,zk) + y
        end
        return y
    else
        return func(x,z)
    end
end

function torch.deepcopy(x)
    return torch.recursiveapply(x,function(v) if torch.isTensor(v) then return v:clone() else return v end end)
end
function torch.deeprandn(x)
    return torch.recursiveapply(x,function(v) return v:randn(v:size()) end)
end
function torch.deepdiff(x,z)
    return torch.recursiveapply(x,function(v,z) return v-z end,z)
end
function torch.deepnorm(x)
    return torch.recursiveapply(x,function(v) if type(v) == 'number' then return v else return v:norm() end end)
end

function torch.deepsum(x,recursion,dim)
	local func
	if dim then
		func = function(v) return v:sum(dim) end
	else
		func = function(v) 
			if type(v) == 'number' then 
				return v 
			else 
				return v:sum() 
			end 
		end
	end
	local recursion = recursion or torch.recursiveapply
	return recursion(x,func)
end

