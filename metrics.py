"""Corpus-level Vietnamese word(space-token) and character error rates."""
from lexicon_utils import normalize_text

def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for x in a:
        curr=[prev[0]+1]
        for j,y in enumerate(b,1):
            curr.append(min(curr[j-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=curr
    return prev[-1]

class ErrorRate:
    def __init__(self):
        self.w_edits=self.w_ref=self.c_edits=self.c_ref=0
        self.count=0
    def add(self, ref, hyp):
        r=normalize_text(ref);h=normalize_text(hyp)
        self.w_edits+=edit_distance(r.split(),h.split());self.w_ref+=len(r.split())
        self.c_edits+=edit_distance(list(r.replace(' ','')),list(h.replace(' ','')))
        self.c_ref+=len(r.replace(' ',''));self.count+=1
    def report(self):
        return dict(wer_space_token=(self.w_edits/self.w_ref if self.w_ref else float('nan')),
                    cer_no_spaces=(self.c_edits/self.c_ref if self.c_ref else float('nan')),
                    utt_count=self.count)
