# agentsearch
Ollama-Firefox-DuckDuckGo based internet search tool to make finding stuff easier

# Basic requirements
You need to have a GPU that can handle the Meta Ollama phi4 language model
You need a basic modern python environment
You need Firefox installed
This has only been tested on Linux, I assume it will work on Mac so long as you have Firefox. I imagine it can be made to work with Windows... I only test on Linux but your LLM of choice can probably help you with other platforms!

# Installation instructions
1. Install Phi4
   $ curl -fsSL https://ollama.com/install.sh | sh # This will download and install the latestest version of ollama from Meta
   $ ollama run phi4 # This will download and start the phi4 model. This is a good test
   $ ollama pull phi4 # This is an optional alternative to the previous command (useful if you already have other models downloaded)
   Phi4 is about a 10 Gb download so maybe start on step 2 in parallel...
2. Download anaconda (https://www.anaconda.com/download)
   Once anaconda is downloaded and installed, create a conda environment and install the base dependencies
   $ conda create -n agent
   $ conda activate agent
   $ conda install selenium httpx tenacity -y
3. If you don't have firefox installed because you are running a special Linux then install it from your package manager
   $ sudo apt update
   $ sudo apt install firefox

# Running agent-based web searches
1. Start the agent from a terminal
   conda activate agent
   python learning-agent.py

--------------- It should look something like this ---------------
   Learning Agent initialized. Enter your question (or 'quit' to exit):

Your question: can you find a recipe for hummus?

Processing your question...
2025-01-11 18:53:49,612 - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
2025-01-11 18:53:49,613 - INFO - Generated search queries: ['Hummus recipe 2023', 'Easy hummus preparation steps 2023', 'Traditional hummus ingredients and method 2023']
2025-01-11 18:53:49,613 - INFO - Generated queries: ['Hummus recipe 2023', 'Easy hummus preparation steps 2023', 'Traditional hummus ingredients and method 2023']
2025-01-11 18:53:50,641 - INFO - Searching DuckDuckGo: https://duckduckgo.com/?q=Hummus+recipe+2023&kl=us-en&k1=-1&atb=v233-1&ia=web
2025-01-11 18:53:57,740 - INFO - Searching DuckDuckGo: https://duckduckgo.com/?q=Easy+hummus+preparation+steps+2023&kl=us-en&k1=-1&atb=v233-1&ia=web
2025-01-11 18:54:02,536 - INFO - Searching DuckDuckGo: https://duckduckgo.com/?q=Traditional+hummus+ingredients+and+method+2023&kl=us-en&k1=-1&atb=v233-1&ia=web
2025-01-11 18:54:25,845 - INFO - HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"

Answer:
Certainly! Here's a comprehensive guide to making traditional, creamy, and delicious homemade hummus by synthesizing information from multiple sources:

### Traditional Hummus Recipe

**Ingredients:**
- **Chickpeas:** 2 cups (cooked or canned) [1][3][5]
- **Tahini:** ⅓ cup [1][6]
- **Lemon Juice:** ¼ to ½ cup, depending on preference for lemony flavor [1][14]
- **Garlic Cloves:** 2-3 cloves, minced or whole [1][8]
- **Salt:** To taste (adjust as needed) [10][11]
- **Water or Chickpea Liquid (Aquafaba):** Start with ¼ cup and adjust for consistency [13][14]
- **Ice Cubes:** Optional, to keep mixture cool during blending [8]

### Instructions:

1. **Prepare the Chickpeas:**
   - If using dried chickpeas, soak overnight and cook until tender (about 2 hours) [13]. Drain them well.
   - For canned chickpeas, drain but reserve some of the liquid for later use to adjust the consistency or reduce lemony flavor if desired [14].

2. **Initial Blend:**
   - In a food processor, combine cooked chickpeas, tahini, minced garlic (or whole cloves), lemon juice, and salt [1][6][8].
   - Add ¼ cup of water or reserved chickpea liquid to help blend the mixture smoothly [13].

3. **Blend to Perfection:**
   - Process for about 40 seconds. Scrape down the sides and bottom of the bowl [8].
   - For extra creaminess, add an ice cube to keep the hummus cool while blending. Continue processing at high speed for another 3 minutes, adding additional ice cubes every 50-60 seconds if needed [1][8].

4. **Adjust Consistency:**
   - If the hummus is too thick, gradually incorporate more water or chickpea liquid until reaching your desired consistency [13].
   - For a less lemony flavor, adjust by reducing lemon juice and increasing reserved chickpea liquid accordingly [14].

5. **Taste and Adjust:**
   - Taste the hummus and adjust seasoning with additional salt or lemon juice as needed [9][11].

6. **Serve:**
   - Transfer the hummus to a serving dish. Optionally, drizzle with olive oil, sprinkle with paprika, or garnish with chopped parsley for extra flavor and presentation.

### Tips:
- Using ice during blending helps maintain texture and creaminess [8].
- Adjust lemon juice according to taste preference—some like it more tangy while others prefer a milder flavor [14].
- For an even smoother hummus, remove the skins from some chickpeas before blending [10].

This recipe combines traditional methods with tips for achieving a creamy, restaurant-style hummus that's perfect for snacking or serving at gatherings. Enjoy your homemade hummus!

References:
[1] Best Hummus Recipe (Plus Tips & Variations) - Cookie and Kate
    https://cookieandkate.com/best-hummus-recipe/
[2] The BEST Hummus Recipe {Easy & Authentic} - Feel Good Foodie
    https://feelgoodfoodie.net/recipe/best-hummus/
[3] The Best Easy Hummus Recipe • It Doesn't Taste Like Chicken
    https://itdoesnttastelikechicken.com/the-best-easy-hummus-recipe/
[4] Authentic Hummus Recipe - The Kitchen Girl
    https://thekitchengirl.com/hummus-recipe/
[5] Best Homemade Hummus (3 Ways!) - The Spice Girl Kitchen
    https://thespicegirlkitchen.com/best-homemade-hummus/
[6] Homemade Hummus in 10 Easy Steps: The Secret Recipe You've Been ...
    https://flavorinsider.com/44764/how-to-make-home-hummus/
[7] How to Make Hummus - Sugar Spun Run
    https://sugarspunrun.com/how-to-make-hummus/
[8] Authentic Homemade Hummus (Quick and Easy Recipe)
    https://www.lemonblossoms.com/blog/authentic-hummus-recipe/
[9] Authentic Hummus Recipe - The Kitchen Girl
    https://thekitchengirl.com/hummus-recipe/
[10] The Best Easy Hummus Recipe - It Doesn't Taste Like Chicken
    https://itdoesnttastelikechicken.com/the-best-easy-hummus-recipe/
[11] The BEST Hummus Recipe {Easy & Authentic} - Feel Good Foodie
    https://feelgoodfoodie.net/recipe/best-hummus/
[12] Authentic Hummus Recipe - The Kitchen Girl
    https://thekitchengirl.com/hummus-recipe/
[13] Easy Hummus Recipe (Authentic and Homemade) - The Mediterranean Dish
    https://www.themediterraneandish.com/how-to-make-hummus/
[14] Hummus Recipe (The Traditional Tasty Way) - Chef Tariq
    https://www.cheftariq.com/recipe/hummus/
[15] Traditional Hummus Recipe: How to Make the Best Hummus Ever!
    https://www.growforagecookferment.com/traditional-hummus-recipe/

Your question: 
-----------------------------
2. Stop the agent at the end by typing "quit" in the question prompt
3. Deactivate the conda environment
   $ conda deactivate
