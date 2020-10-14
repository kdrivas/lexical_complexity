## Findings

- In the table below, I find Multi has more complexity than Single

|          	| Single   	|          	| Multi    	|          	|
|----------	|----------	|----------	|----------	|----------	|
| corpus   	| mean     	| std      	| mean     	| std      	|
| bible    	| 0.290956 	| 0.131206 	| 0.375036 	| 0.150999 	|
| biomed   	| 0.324377 	| 0.151944 	| 0.493693 	| 0.171375 	|
| europarl 	| 0.286996 	| 0.108626 	| 0.382457 	| 0.113062 	|

- The table below show the count per number of word occurence, is this table suggest the 0.8 of mae is a higher error? -> Look at the mean of std group

| # Occurences 	| count 	| mean of mean group 	| mean of std group 	|
|--------------	|-------	|--------------------	|-------------------	|
| 1            	| 1704  	| 0.323023           	| NaN               	|
| 2            	| 578   	| 0.310819           	| 0.054978          	|
| 3            	| 338   	| 0.290813           	| 0.061015          	|
| 4            	| 207   	| 0.297613           	| 0.061420          	|
| 5            	| 485   	| 0.288763           	| 0.064828          	|
| 6            	| 13    	| 0.271687           	| 0.075082          	|
| 7            	| 2     	| 0.234665           	| 0.065810          	|
| 10           	| 1     	| 0.365420           	| 0.055241          	|