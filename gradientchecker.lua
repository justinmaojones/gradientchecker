require 'nn';
require 'recursiveUtils';


local deepdot = function(x,z)
    --return torch.recursiveapply(x,function(v,z) return torch.sum(torch.cmul(v,z)) end,z)
    local _deepdot
    _deepdot = function(v,z)
        local m = v:nDimension()
        if m == 1 then
            return torch.dot(v,z)
        else
			return torch.cmul(v,z):sum()
		end
    end
	return torch.recursiveaccumulate(x,_deepdot,z)
end


function nn.gradientapproximator(checkvars,inputs,model,criterion,target)
    local gradapprox
    gradapprox = function(v)
		local approxGradInput
		if v:type() == 'torch.LongTensor' then
			approxGradInput = torch.LongTensor(0)
		else
			local m = v:nDimension()
			approxGradInput = torch.zeros(v:size())
			local n = v:size(1)
			if m > 1 then
				for i = 1,n do
					approxGradInput:select(1,i):copy(gradapprox(v:select(1,i)))
				end
			else
				local v0 = v:clone()
				local eps = 1e-4
				local perturb
				perturb = function(i,eps)
					v:copy(v0)
					v[i] = v[i] + eps
				end
				for i = 1, n do
					perturb(i,eps)
					local output = model:forward(inputs)
					if criterion then
						output = criterion:forward(output,target)
					end
					local yp = torch.deepsum(output,torch.recursiveaccumulate)
					perturb(i,-eps)
					local output = model:forward(inputs)
					if criterion then
						output = criterion:forward(output,target)
					end
					local ym = torch.deepsum(output,torch.recursiveaccumulate)
					if m == 1 then
						local val = (yp-ym)/(2*eps)
						approxGradInput[i] = (yp-ym)/(2*eps)
					else
						approxGradInput:select(m,i):copy((yp-ym)/(2*eps))
					end
				end
				-- reset
				v:copy(v0)
			end
		end
        return approxGradInput
    end
    return torch.recursiveapply(checkvars,gradapprox)
end

--local eraseEmptyTensors
eraseEmptyTensors = function(x,z)
	if type(x) == 'table' then
		local yx = {}
		local yz = z and {} 
		for k,v in pairs(x) do
			local xk,zk = eraseEmptyTensors(v,z and z[k])
			yx[k] = xk
			yz[k] = z and zk
		end
		return yx,yz
	else
		if x:nElement() == 0 then
			print('warning: empty tensor encountered')
			return nil,nil
		else
			return x,z
		end
	end
end

function nn.gradientchecker(input,model,criterion,target)
	-- criterion and target are optional
	local p,dp = model:getParameters()
	dp:zero()
	local output = torch.deepcopy(model:forward(input))
	local loss
	if criterion then
		print('--check includes criterion')
		 loss = criterion:forward(output,target)
	end
	local gradOutput
	if criterion then
		gradOutput = criterion:backward(output,target)
	else
		gradOutput = torch.deepcopy(output)
		gradOutput = torch.recursiveapply(gradOutput,function(v) return v:fill(1) end)
	end

	-- check gradInput
	local gradInput = torch.deepcopy(model:backward(input,gradOutput))
	local approxGradInput = nn.gradientapproximator(input,input,model,criterion,target)
	--print('gradInput[1][2]',gradInput[1][2])
	--print('approxGradInput[1][2]',approxGradInput[1][2])
	--print('gradInput',gradInput)
	--print('approxGradInput',approxGradInput)
	gradInput,approxGradInput = eraseEmptyTensors(gradInput,approxGradInput)
	--print('gradInput',gradInput)
	--print('approxGradInput',approxGradInput)
	local gradInputCheck = torch.deepnorm(torch.deepdiff(gradInput,approxGradInput))
	print('gradInputCheck = ',gradInputCheck)

	if p:nElement() > 0 then
		-- check gradParameters
		local approxGradParams = nn.gradientapproximator(p,input,model,criterion,target)
		local gradParamCheck = torch.deepnorm(torch.deepdiff(dp,approxGradParams))
		print('gradParamCheck = ',gradParamCheck)
	else
		print('no gradParams to check')
	end
end

function nn.gradientchecker2(input,model,checkInput)
	-- criterion and target are optional
	local p,dp = model:getParameters()
	dp:zero()
	local output = torch.deepcopy(model:forward(input))
	local loss
	if criterion then
		 loss = criterion:forward(output,target)
	end
	local gradOutput
	if criterion then
		gradOutput = criterion:backward(output,target)
	else
		gradOutput = torch.deepcopy(output)
		gradOutput = torch.recursiveapply(gradOutput,function(v) return v:fill(1) end)
	end

	-- check gradInput
	local gradInput = torch.deepcopy(model:backward(input,gradOutput))
	local approxGradInput = nn.gradientapproximator(input,input,model,criterion,target)
	print('gradInput[1][2]',gradInput[1][2])
	print('gradInput',gradInput)
	print('approxGradInput',approxGradInput)
	gradInput,approxGradInput = eraseEmptyTensors(gradInput,approxGradInput)
	print('gradInput',gradInput)
	print('approxGradInput',approxGradInput)
	local gradInputCheck = torch.deepnorm(torch.deepdiff(gradInput,approxGradInput))
	print('gradInputCheck = ',gradInputCheck)

	if p:nElement() > 0 then
		-- check gradParameters
		local approxGradParams = nn.gradientapproximator(p,input,model,criterion,target)
		local gradParamCheck = torch.deepnorm(torch.deepdiff(dp,approxGradParams))
		print('gradParamCheck = ',gradParamCheck)
	else
		print('no gradParams to check')
	end
end

