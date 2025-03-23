# Loss Algorithm

Cc1c(Cl)cccc1Brc1ncccc1C(=O)OCC(O)CO

Each swap = .1
Each letter = 28*.01
Each Sequence = [EOS]-[BOS]*(28 * .1)
Each Sanitize = 1
Invalid Smiles = 2 

Total Loss = (28*.1)12 - (.1*10)

Loss = 0 or positive

No redundant information or if you do reward in some way by reducing the loss

Unless its perfect, no 0 loss

No negative loss at all

loss = 1 - 15*.01 - 2*.01

What does it mean to be close?

Br -> Next biggest/smallest in atomic weight ->  -?  ->  +
Find a metric of closeness for swaps based on chemistry
Proton count
Valence count / how badly they want to shed or gain an electron
carbon like carbon

Physics informed neural network


Loss = criteon(y,data) + (dx/dy - x~ + awdgw)**2