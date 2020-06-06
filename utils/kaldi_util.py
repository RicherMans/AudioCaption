
def parsecopyfeats(feat, cmvn=False, delta=False, splice=None):
    outstr = "copy-feats ark:{} ark:- |".format(feat)
    if cmvn:
        outstr += "apply-cmvn-sliding --center ark:- ark:- |"
    if delta:
        outstr += "add-deltas ark:- ark:- |"
    if splice and splice > 0:
        outstr += "splice-feats --left-context={} --right-context={} ark:- ark:- |".format(
            splice, splice)
    return outstr

