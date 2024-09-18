This was mostly a learning project to get my feet wet with doing regression with pytorch. 

The goal was to see if I could create a model to take some simple data (college stats and combine measurements) and predict numbers for rookie NFL receivers. There are a few problems with this:

- The data is pretty limited for receivers that played in a modern offensive environment. Numbers from 2000 look a lot different than numbers now, so there are only really 15ish years of numbers that can reasonably be used
- The data used for input is not sufficient. Things like role in the offense, what school they played at, etc. are not accounted for in those numbers. I wanted to see if a super simple model would be sufficient, and the answer is no
- Finding this data is not easy. I did my best, but had to grab data from multiple sources and put them together by comparing names. Different sources can have different spellings or sometimes use nickname vs legal name. Combine data is often missing certain measurements
- Number of games played in a player's rookie season is highly variable, and injuries can also play into things. A player's rookie numbers are very sensitive to things that have nothing to do with his ability

The end result was that any tweaking I did to update the model size, learning rate, decay, etc. turned out to be futile. Any decreases in training loss were proven to be strictly memorization of the training data by the validation.

If it really was easy enough that I could put together an accurate model real quick while messing around in my free time, I guess NFL GM's wouldn't get paid so much :)
  
