# Wheat Coverage Hypothesis

Discovery uses previously designated BUILD seed 990057, not official canary
980003. The official v2 result is retained as rejection evidence only.

At step 187, a weed blocks planting for unit 6 on seat 1. By step 244, another
worker requests five wheat but only three are available, versus five on seat 0.
At steps 257 and 261, its FEED actions find no carried wheat. At step 375 the
same scheduled wool liquidation produces a 968-point additional money deficit.
These observations motivate a supply-shortfall repair, not a unit tape rewrite.
They do not by themselves prove that the weed is the sole cause of every loss.

A small Python/C++-engine causal probe purchases the projected deficit before
the next own scheduled wheat pickup, preserving worker actions and parent state.
An unrestricted version flips the known loss but loses both seats on BUILD
990031 by 17 points. Restricting purchases to steps 144 through 646 wins all
eight diagnostic games, including +2128 on the known losing case and +613 on
both seats of 990031. This is a discovery result, not an acceptance gate.

The native search tests nine bounded setup/end windows on disjoint fresh BUILD
1020000-1020063 plus the original diagnostic BUILD screen. Runtime reads only
own projected stock and the current own route. No opponent identity, seed,
fingerprint, or hidden information is used. No parent assets are modified.
