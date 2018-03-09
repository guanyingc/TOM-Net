local AGG, parent = torch.class('nn.AffineGridGeneratorUSOF', 'nn.Module')

--[[
   alternate version of AffineGridGeneratorBHWD(height, width) for optical flow:

   AffineGridGeneratorBHWD:updateOutput(transformMatrix)
   AffineGridGeneratorBHWD:updateGradInput(transformMatrix, gradGrids)
   AffineGridGeneratorBHWD will take 2x3 an affine image transform matrix (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will result in an affine transform.
   AffineGridGenerator 
   - takes (B,2,3)-shaped transform matrices as input (B=batch).
   - outputs a grid in BHWD layout, that can be used directly with BilinearSamplerBHWD
   - initialization of the previous layer should biased towards the identity transform :
      | 1  0  0 |
      | 0  1  0 |
   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
]]

function AGG:__init(sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
end

local function addOuterDim(t)
    local sizes    = t:size()
    local newsizes = torch.LongStorage(sizes:size()+1)
    newsizes[1] = 1
    for i = 1,sizes:size() do
        newsizes[i+1] = sizes[i]
    end
    return t:view(newsizes)
end

function AGG:updateOutput(_flows)
    local flows  = _flows:clone()

    local batch  = flows:size(1)
    local height = flows:size(3)
    local width  = flows:size(4)
    local _baseGrid
    local baseGrid 
    if flows:type() == 'torch.CudaTensor' then
        _baseGrid = torch.CudaTensor(batch, height, width, 2)
        baseGrid  = torch.CudaTensor(height, width, 2) -- H*(v, u)
    else
        _baseGrid = torch.Tensor(batch, height, width, 2)
        baseGrid  = torch.Tensor(height, width, 2) -- H*(v, u)
    end
    for i=1, height do
        baseGrid:select(3,1):select(1,i):fill(-1 + (i-1)/(height-1) * 2)
    end
    for j=1, width do
        baseGrid:select(3,2):select(2,j):fill(-1 + (j-1)/(width-1) * 2)
    end
    for k = 1, batch do
        _baseGrid:select(1,k):copy(baseGrid)
    end

    assert(flows:nDimension()==4 , 'dimension of flow vectors has to be (bx2xhxw)')

    if flows:size(2) == 2 then
        flows = flows:transpose(2,3):transpose(3,4)
    end
    
    self.output:resize(batch, height, width, 2)
    flows[{ {},{},{},{1} }]:div(height/2) -- v
    flows[{ {},{},{},{2} }]:div(width/2)  -- u

    self.output = torch.add(_baseGrid , flows)
    return self.output
end

function AGG:updateGradInput(_flows, _gradGrid)
   local flows
   if _flows:size(2) == 2 then
       flows = _flows:transpose(2,3):transpose(3,4)
   else
       flows = _flows
   end
   local batch  = flows:size(1)
   local height = flows:size(2)
   local width  = flows:size(3) 

   self.gradInput = _gradGrid:clone()
   self.gradInput[{ {},{},{},{1} }]:div(height/2)
   self.gradInput[{ {},{},{},{2} }]:div(width/2)
   self.gradInput = self.gradInput:transpose(3,4):transpose(2,3)
   return self.gradInput
end
