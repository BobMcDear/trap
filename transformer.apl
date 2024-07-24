⎕IO←0

:Namespace TRANSFORMER

⍝ Model, training, and optimizer hyperparameters
ITERS←500 ⋄ BS←16 ⋄ VCB←128 ⋄ SEQ←32 ⋄ DPTH←4 ⋄ HEADS←4 ⋄ DIM←128
LR←3E¯4 ⋄ BETA1←0.9 ⋄ BETA2←0.999 ⋄ WD←1E¯2

⍝ Miscellaneous
MASK←-1E10×~(⍳SEQ)∘.≥⍳SEQ
GELUF←(2÷○1)*0.5

⍝ Parameter initialization functions
RND←{d←10*9 ⋄ (x y)←⊂[1+⍳⍴,⍵](?(2,⍵)⍴d)÷d ⋄ ((¯2×⍟x)*0.5)×1○○2×y} ⍝ Samples from normal distribution
LIN←{WB←↑{(0.02×RND ⍵) ((1↓⍵)⍴0)}¨⍺⍴⊂⍵ ⋄ (WB[;0]) (WB[;1])} ⍝ Initiliazes weights & biases

⍝ Model parameters
WTE←⊃⊃1 LIN VCB DIM ⋄ WPE←⊃⊃1 LIN SEQ DIM ⋄ WH←⊃⊃1 LIN DIM VCB
WLN←(1+2×DPTH)⍴⊂DIM⍴1 ⋄ BLN←(1+2×DPTH)⍴⊂DIM⍴0
WQKV BQKV←DPTH LIN DIM HEADS (DIM÷HEADS) 3 ⋄ WO BO←DPTH LIN DIM DIM
W1 B1←DPTH LIN DIM (4×DIM) ⋄ W2 B2←DPTH LIN (4×DIM) DIM

⍝ Activation tensors for backpropagation
W2INP←ACTINP←W1INP←WOINP←ATTN←V←K←Q←WQKVINP←DPTH⍴⊂WHINP←WTEINP←⍬
STD←MEAN←LNINP←(1+2×DPTH)⍴⊂⍬
CETARG←CEPROB←⍬

⍝ Gradients
∆B2←∆W2←∆B1←∆W1←∆BO←∆WO←∆BQKV←∆WQKV←DPTH⍴⊂∆WH←∆WPE←∆WTE←⍬
∆BLN←∆WLN←(1+2×DPTH)⍴⊂⍬

⍝ Optimizer states
T←0
M1WH←M1B2←M1W2←M1B1←M1W1←M1BO←M1WO←M1BQKV←M1WQKV←M1BLN←M1WLN←M1WPE←M1WTE←0
M2WH←M2B2←M2W2←M2B1←M2W1←M2BO←M2WO←M2BQKV←M2WQKV←M2BLN←M2WLN←M2WPE←M2WTE←0

⍝ Utilities
UNSQZ←{((⍴⍵),1)⍴⍵} ⍝ Inserts axis of size 1 as last dimension
AVG←{UNSQZ (+/⍵)÷¯1↑⍴⍵} ⍝ Averges along last dimension
SM←{exp←*⍵-⍤1⊢UNSQZ⌈/⍵ ⋄ exp÷⍤1⊢UNSQZ+/exp} ⍝ Applies softmax along last dimension

FWD←{
    ⍝ Gets token embeddings
    TE←{WTE[WTEINP⊢←⍵;]}

    ⍝ Adds position embeddings to input
    PE←{WPE[⍳1⌷⍴⍵;]+⍤2⊢⍵}

    ⍝ Layer-normalizes input
    LN←{
        diff←⍵-⍤1⊢⊃MEAN[⍺]←⊂AVG ⊃LNINP[⍺]←⊂⍵
        (⍺⊃BLN)+⍤1⊢(⍺⊃WLN)×⍤1⊢diff÷⍤1⊢⊃STD[⍺]←⊂(1E¯5+AVG diff*2)*0.5
    }

    ⍝ Applies multi-headed self-attention to input
    MHSA←{
        qkv←(⍺⊃BQKV)+⍤3⊢(⊃WQKVINP[⍺]←⊂⍵)+.×⍺⊃WQKV
        q k v←{0 2 1 3⍉⍵⌷⍤1⊢qkv}¨⍳3 ⋄ Q[⍺]←⊂q ⋄ K[⍺]←⊂k ⋄ V[⍺]←⊂v
        ATTN[⍺]←⊂attn←SM MASK[⍳2⌷⍴q;⍳2⌷⍴k]+⍤2⊢(q+.×⍤2⊢⍉⍤2⊢k)÷(¯1↑⍴k)*0.5
        (⍺⊃BO)+⍤1⊢(⊃WOINP[⍺]←⊂(⍴⍵)⍴0 2 1 3⍉attn+.×⍤2⊢v)+.×⍺⊃WO
    }

    ⍝ Transforms input using multilayer perceptron with one hidden layer
    MLP←{
        ACTINP[⍺]←⊂h←(⍺⊃B1)+⍤1⊢(⊃W1INP[⍺]←⊂⍵)+.×⍺⊃W1
        (⍺⊃B2)+⍤1⊢(⊃W2INP[⍺]←⊂0.5×h×1+7○GELUF×h+0.044715×h*3)+.×⍺⊃W2
    }

    ⍝ Passes input through a transformer block
    BLK←{
        ind inp←⍵
        out←inp+ind MHSA (2×ind) LN inp
        (ind+1) (out+ind MLP (1+2×ind) LN out)
    }

    ⍝ Produces next token predictions
    HEAD←{(WHINP⊢←(2×DPTH) LN ⍵)+.×WH}

    ⍝ Calculates cross-entropy loss
    CE←{{(+⌿⍵)÷≢⍵},(UNSQZ CETARG⊢←⍺)⌷⍤1⊢-⍟CEPROB⊢←SM ⍵}

    out←HEAD 1⊃BLK⍣DPTH⊢0 (PE TE ⍵)
    ⍺←⍬ ⋄ 0=≢⍺:out ⋄ ⍺ CE out
}

BWD←{
    ∆TE←{∆WTE⊢←(⍴WTE)⍴0 ⋄ ∆WTE[WTEINP;]+←⍵}

    ∆PE←{∆WPE⊢←(⍴WPE)⍴0 ⋄ ∆WPE[⍳1⌷⍴⍵;]←+⌿⍵ ⋄ ⍵}

    ∆LN←{
        ∆WLN[⍺]←⊂+⌿,[⍳2]⍵×prelin←((⍺⊃LNINP)-⍤1⊢⍺⊃MEAN)÷⍤1⊢⍺⊃STD ⋄ ∆BLN[⍺]←⊂+⌿,[⍳2]⍵
        wgprod←⍵×⍤1⊢⍺⊃WLN
        (wgprod-(prelin×⍤1⊢AVG prelin×wgprod)+⍤1⊢AVG wgprod)÷⍤1⊢⍺⊃STD
    }

    ∆MHSA←{
        ∆WO[⍺]←⊂(⍉,[⍳2]⍺⊃WOINP)+.×,[⍳2]⍵ ⋄ ∆BO[⍺]←⊂+⌿,[⍳2]⍵
        ∆woinp←0 2 1 3⍉((2↑⍴⍵),HEADS (DIM÷HEADS))⍴⍵+.×⍉⍺⊃WO
        ∆v←(⍉⍤2⊢⍺⊃ATTN)+.×⍤2⊢∆woinp
        ∆qkprod←(⍺{(⍺⊃ATTN)×⍵-⍤1⊢UNSQZ +/⍵×⍺⊃ATTN}∆woinp+.×⍤2⊢⍉⍤2⊢⍺⊃V)÷(¯1↑⍴⍺⊃K)*0.5
        ∆k←⍉⍤2⊢(⍉⍤2⊢⍺⊃Q)+.×⍤2⊢∆qkprod
        ∆q←∆qkprod+.×⍤2⊢⍺⊃K
        ∆qkv←4 0 2 1 3⍉↑∆q ∆k ∆v
        ∆WQKV[⍺]←⊂(⍉,[⍳2]⍺⊃WQKVINP)+.×,[⍳2]∆qkv ⋄ ∆BQKV[⍺]←⊂+⌿,[⍳2]∆qkv
        (,[2+⍳3]∆qkv)+.×⍉,[1+⍳3]⍺⊃WQKV
    }

    ∆MLP←{
        ∆W2[⍺]←⊂(⍉,[⍳2]⍺⊃W2INP)+.×,[⍳2]⍵ ⋄ ∆B2[⍺]←⊂+⌿,[⍳2]⍵
        ∆w2inp←⍵+.×⍉⍺⊃W2
        arg←GELUF×(⍺⊃ACTINP)+0.044715×(⍺⊃ACTINP)*3
        ∆h←∆w2inp×(0.5×1+7○arg)+(⍺⊃ACTINP)×0.5×(÷(6○arg)*2)×GELUF×1+0.134145×(⍺⊃ACTINP)*2
        ∆W1[⍺]←⊂(⍉,[⍳2]⍺⊃W1INP)+.×,[⍳2]∆h ⋄ ∆B1[⍺]←⊂+⌿,[⍳2]∆h
        ∆h+.×⍉⍺⊃W1
    }

    ∆BLK←{
        ind ∆out←⍵
        ∆inp←∆out+(1+2×ind) ∆LN ind ∆MLP ∆out
        (ind-1) (∆inp+(2×ind) ∆LN ind ∆MHSA ∆inp)
    }

    ∆HEAD←{∆WH⊢←(⍉,[⍳2]WHINP)+.×,[⍳2]⍵ ⋄ (2×DPTH) ∆LN ⍵+.×⍉WH}

    ∆CE←{(CEPROB-CETARG∘.=⍳¯1↑⍴CEPROB)÷×/⍴CETARG}

    _←∆TE ∆PE 1⊃∆BLK⍣DPTH⊢(DPTH-1) (∆HEAD ∆CE ⍬) ⋄ ⍬
}

TRAIN←{
    ⍝ Updates set of parameters using AdamW
    OPT←{
        P ∆P M1 M2←⍵
        M1←(BETA1×M1)+∆P×1-BETA1 ⋄ M2←(BETA2×M2)+∆P×∆P×1-BETA2
        (P-LR×(WD×P)+(M1÷1-BETA1*T)÷1E¯8+(M2÷1-BETA2*T)*0.5) M1 M2
    }

    ⍝ Performs one training iteration
    ITER←{
        seq←data[(?BS⍴(⍴data)-SEQ+1)∘.+⍳SEQ+1] ⋄ inp←¯1↓⍤1⊢seq ⋄ targ←1↓⍤1⊢seq
        loss←targ FWD inp ⋄ _←BWD ⍬ ⋄ T+←1
        X←OPT WTE ∆WTE M1WTE M2WTE ⋄ WTE⊢←0⊃X ⋄ M1WTE⊢←1⊃X ⋄ M2WTE⊢←2⊃X
        X←OPT WPE ∆WPE M1WPE M2WPE ⋄ WPE⊢←0⊃X ⋄ M1WPE⊢←1⊃X ⋄ M2WPE⊢←2⊃X
        X←OPT WLN ∆WLN M1WLN M2WLN ⋄ WLN⊢←0⊃X ⋄ M1WLN⊢←1⊃X ⋄ M2WLN⊢←2⊃X
        X←OPT BLN ∆BLN M1BLN M2BLN ⋄ BLN⊢←0⊃X ⋄ M1BLN⊢←1⊃X ⋄ M2BLN⊢←2⊃X
        X←OPT WQKV ∆WQKV M1WQKV M2WQKV ⋄ WQKV⊢←0⊃X ⋄ M1WQKV⊢←1⊃X ⋄ M2WQKV⊢←2⊃X
        X←OPT BQKV ∆BQKV M1BQKV M2BQKV ⋄ BQKV⊢←0⊃X ⋄ M1BQKV⊢←1⊃X ⋄ M2BQKV⊢←2⊃X
        X←OPT WO ∆WO M1WO M2WO ⋄ WO⊢←0⊃X ⋄ M1WO⊢←1⊃X ⋄ M2WO⊢←2⊃X
        X←OPT BO ∆BO M1BO M2BO ⋄ BO⊢←0⊃X ⋄ M1BO⊢←1⊃X ⋄ M2BO⊢←2⊃X
        X←OPT W1 ∆W1 M1W1 M2W1 ⋄ W1⊢←0⊃X ⋄ M1W1⊢←1⊃X ⋄ M2W1⊢←2⊃X
        X←OPT B1 ∆B1 M1B1 M2B1 ⋄ B1⊢←0⊃X ⋄ M1B1⊢←1⊃X ⋄ M2B1⊢←2⊃X
        X←OPT W2 ∆W2 M1W2 M2W2 ⋄ W2⊢←0⊃X ⋄ M1W2⊢←1⊃X ⋄ M2W2⊢←2⊃X
        X←OPT B2 ∆B2 M1B2 M2B2 ⋄ B2⊢←0⊃X ⋄ M1B2⊢←1⊃X ⋄ M2B2⊢←2⊃X
        X←OPT WH ∆WH M1WH M2WH ⋄ WH⊢←0⊃X ⋄ M1WH⊢←1⊃X ⋄ M2WH⊢←2⊃X
        ⍬
    }

    data←⍵
    _←ITER⍣ITERS⊢⍬
}

⍝ Greedily generates next tokens
GEN←{{⍵,(⊃⍒)⍤1⊢¯1↑[1]FWD (-SEQ⌊1⌷⍴⍵)↑[1]⍵}⍣⍺⊢⍵}

:EndNamespace
