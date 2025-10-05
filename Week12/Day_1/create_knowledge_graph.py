import networkx as nx
import matplotlib.pyplot as plt

# Sample data from IMDB

data = [
    {"Rank": "1", "Title": "Guardians of the Galaxy", "Genre": "Action,Adventure,Sci-Fi", "Director": "James Gunn", "Actors": "Chris Pratt, Vin Diesel, Bradley Cooper, Zoe Saldana", "Rating": "8.1", "Revenue (Millions)": "333.13"},
    {"Rank": "2", "Title": "Prometheus", "Genre": "Adventure,Mystery,Sci-Fi", "Director": "Ridley Scott", "Actors": "Noomi Rapace, Logan Marshall-Green, Michael Fassbender, Charlize Theron", "Rating": "7", "Revenue (Millions)": "126.46"},
    {"Rank": "3", "Title": "Split", "Genre": "Horror,Thriller", "Director": "M. Night Shyamalan", "Actors": "James McAvoy, Anya Taylor-Joy, Haley Lu Richardson, Jessica Sula", "Rating": "7.3", "Revenue (Millions)": "138.12"},
    {"Rank": "4", "Title": "Sing", "Genre": "Animation,Comedy,Family", "Director": "Christophe Lourdelet", "Actors": "Matthew McConaughey,Reese Witherspoon, Seth MacFarlane, Scarlett Johansson", "Rating": "7.2", "Revenue (Millions)": "270.32"},
    {"Rank": "5", "Title": "Suicide Squad", "Genre": "Action,Adventure,Fantasy", "Director": "David Ayer", "Actors": "Will Smith, Jared Leto, Margot Robbie, Viola Davis", "Rating": "6.2", "Revenue (Millions)": "325.02"},
    # More data...
]

# Create a directed graph
G = nx.Graph()

# Add nodes and edges
for movie in data:
    title = movie['Title']
    director = movie['Director']
    genres = movie['Genre'].split(',')
    actors = movie['Actors'].split(', ')
    rating = float(movie['Rating'])
    revenue = float(movie['Revenue (Millions)']) if movie['Revenue (Millions)'] else 0
    
    # Movie node
    G.add_node(title, type='movie')
    
    # Director node
    G.add_node(director, type='director')
    G.add_edge(title, director, weight=revenue)
    
    # Genre nodes
    for genre in genres:
        G.add_node(genre, type='genre')
        G.add_edge(title, genre, weight=rating)
    
    # Actor nodes
    for actor in actors:
        G.add_node(actor, type='actor')
        G.add_edge(title, actor, weight=rating)

# Draw the graph
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw(G, pos, with_labels=True, node_size=20, font_size=8)
plt.savefig("knowledge_graph.png")

# Save the graph as GraphML
nx.write_graphml(G, "movie_kg.graphml")

# Output the path to the saved graph image
result = "knowledge_graph.png"