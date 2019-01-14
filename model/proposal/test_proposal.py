from model.proposal.proposal_layer import generate_anchors
out = generate_anchors(16, 60,[1,0.5,2], [64,128])
print(out.shape)