"""Sample PokerStars hand history text blocks for testing."""

# A complete 6-max No Limit Hold'em hand with all streets.
SAMPLE_HAND_FULL = """\
PokerStars Hand #12345: Hold'em No Limit ($1/$2) - 2024/01/15 12:00:00 ET
Table 'TableName' 6-max Seat #3 is the button
Seat 1: Player1 ($200 in chips)
Seat 2: Player2 ($185.50 in chips)
Seat 3: Player3 ($150 in chips)
Seat 4: Player4 ($220 in chips)
Seat 5: Player5 ($175 in chips)
Seat 6: Hero ($300 in chips)
Player4: posts small blind $1
Player5: posts big blind $2
*** HOLE CARDS ***
Dealt to Hero [Ah Kd]
Hero: raises $4 to $6
Player1: folds
Player2: calls $6
Player3: folds
Player4: folds
Player5: calls $4
*** FLOP *** [Js 7h 2c]
Player5: checks
Hero: bets $10
Player2: calls $10
Player5: folds
*** TURN *** [Js 7h 2c] [9d]
Hero: bets $25
Player2: calls $25
*** RIVER *** [Js 7h 2c 9d] [3s]
Hero: bets $50
Player2: folds
*** SUMMARY ***
Total pot $85 | Rake $2
"""

# A hand that ends preflop (everyone folds to a raise).
SAMPLE_HAND_PREFLOP_ONLY = """\
PokerStars Hand #12346: Hold'em No Limit ($1/$2) - 2024/01/15 12:05:00 ET
Table 'TableName' 6-max Seat #4 is the button
Seat 1: Player1 ($200 in chips)
Seat 3: Player3 ($150 in chips)
Seat 4: Player4 ($220 in chips)
Seat 5: Player5 ($175 in chips)
Seat 6: Hero ($306 in chips)
Player5: posts small blind $1
Hero: posts big blind $2
*** HOLE CARDS ***
Dealt to Hero [Ts Td]
Player1: raises $4 to $6
Player3: folds
Player4: folds
Player5: folds
Hero: raises $14 to $20
Player1: folds
*** SUMMARY ***
Total pot $13 | Rake $0
"""

# A hand with a showdown.
SAMPLE_HAND_SHOWDOWN = """\
PokerStars Hand #12347: Hold'em No Limit ($1/$2) - 2024/01/15 12:10:00 ET
Table 'TestTable' 6-max Seat #1 is the button
Seat 1: Player1 ($200 in chips)
Seat 2: Player2 ($150 in chips)
Seat 4: Hero ($300 in chips)
Player2: posts small blind $1
Hero: posts big blind $2
*** HOLE CARDS ***
Dealt to Hero [Qh Qd]
Player1: raises $4 to $6
Player2: calls $5
Hero: raises $14 to $20
Player1: calls $14
Player2: folds
*** FLOP *** [8c 5d 2h]
Hero: bets $30
Player1: calls $30
*** TURN *** [8c 5d 2h] [Ks]
Hero: checks
Player1: bets $40
Hero: calls $40
*** RIVER *** [8c 5d 2h Ks] [3c]
Hero: checks
Player1: checks
*** SHOW DOWN ***
Hero: shows [Qh Qd]
Player1: shows [Jh Jd]
Hero collected $183 from pot
*** SUMMARY ***
Total pot $185 | Rake $2
"""

# A hand with all-in action.
SAMPLE_HAND_ALL_IN = """\
PokerStars Hand #12348: Hold'em No Limit ($1/$2) - 2024/01/15 12:15:00 ET
Table 'AllInTable' 6-max Seat #2 is the button
Seat 1: Player1 ($50 in chips)
Seat 2: Player2 ($200 in chips)
Seat 5: Hero ($300 in chips)
Player1: posts small blind $1
Hero: posts big blind $2
*** HOLE CARDS ***
Dealt to Hero [As Ac]
Player2: raises $4 to $6
Player1: raises $44 to $50 and is all-in
Hero: calls $48
Player2: calls $44
*** FLOP *** [Kh Td 4s]
Hero: checks
Player2: checks
*** TURN *** [Kh Td 4s] [7c]
Hero: checks
Player2: checks
*** RIVER *** [Kh Td 4s 7c] [2d]
Hero: checks
Player2: checks
*** SUMMARY ***
Total pot $150 | Rake $3
"""

# Two hands separated by blank lines (for multi-hand parsing tests).
SAMPLE_MULTI_HANDS = (
    SAMPLE_HAND_PREFLOP_ONLY + "\n\n\n" + SAMPLE_HAND_SHOWDOWN
)
