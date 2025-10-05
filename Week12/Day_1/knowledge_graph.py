import networkx as nx
import matplotlib.pyplot as plt

# Data obtained from the IMDb dataset
movies = [
    {"title": "Guardians of the Galaxy", "director": "James Gunn", "genres": ["Action", "Adventure", "Sci-Fi"], "actors": ["Chris Pratt", "Vin Diesel", "Bradley Cooper", "Zoe Saldana"], "rating": 8.1, "revenue": 333.13},
    {"title": "Prometheus", "director": "Ridley Scott", "genres": ["Adventure", "Mystery", "Sci-Fi"], "actors": ["Noomi Rapace", "Logan Marshall-Green", "Michael Fassbender", "Charlize Theron"], "rating": 7.0, "revenue": 126.46},
    {"title": "Split", "director": "M. Night Shyamalan", "genres": ["Horror", "Thriller"], "actors": ["James McAvoy", "Anya Taylor-Joy", "Haley Lu Richardson", "Jessica Sula"], "rating": 7.3, "revenue": 138.12},
    {"title": "Sing", "director": "Christophe Lourdelet", "genres": ["Animation", "Comedy", "Family"], "actors": ["Matthew McConaughey", "Reese Witherspoon", "Seth MacFarlane", "Scarlett Johansson"], "rating": 7.2, "revenue": 270.32},
    {"title": "Suicide Squad", "director": "David Ayer", "genres": ["Action", "Adventure", "Fantasy"], "actors": ["Will Smith", "Jared Leto", "Margot Robbie", "Viola Davis"], "rating": 6.2, "revenue": 325.02}
]

G = nx.Graph()

# Add nodes and edges
def add_movie_data(movie):
    G.add_node(movie['title'], type='Movie')
    G.add_node(movie['director'], type='Director')
    G.add_edge(movie['title'], movie['director'], weight=movie['rating'])

    for genre in movie['genres']:
        G.add_node(genre, type='Genre')
        G.add_edge(movie['title'], genre, weight=movie['rating'])

    for actor in movie['actors']:
        G.add_node(actor, type='Actor')
        G.add_edge(movie['title'], actor, weight=movie['rating'])


for movie in movies:
    add_movie_data(movie)

# Draw the graph
pos = nx.spring_layout(G, k=0.5, iterations=50)
plt.figure(figsize=(12, 12))

# Define colors based on node types
def node_color(node_type):
    return {'Movie': 'blue', 'Director': 'green', 'Actor': 'red', 'Genre': 'orange'}.get(node_type, 'grey')

# Node color and size
colors = [node_color(G.nodes[node]['type']) for node in G.nodes()]
sizes = [300 if G.nodes[node]['type'] == 'Movie' 
         else 200 if G.nodes[node]['type'] == 'Director' 
         else 100 for node in G.nodes()]

nx.draw(G, pos, with_labels=True, node_color=colors, node_size=sizes, font_size=8, font_weight='bold', edge_color='black')
plt.title("Movie Knowledge Graph")
plt.savefig("movie_knowledge_graph.png")
plt.show()