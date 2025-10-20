// frontend/src/App.jsx
import React, { useState } from "react";
import axios from "./api";
import ProductCard from "./components/ProductCard";
import SearchBar from "./components/SearchBar";
import "./styles.css";


function App() {
  const [query, setQuery] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [hero, setHero] = useState("");

  const handleSearch = async () => {
    if (!query.trim()) return;
    const res = await axios.post("/recommend", { query, image_url: imageUrl });
    setHero(res.data.hero);
    setRecommendations(res.data.recommendations || []);
  };

  return (
    <div className="app">
      <h1>🪑 AI Furniture Recommender</h1>
      <SearchBar
        query={query}
        setQuery={setQuery}
        imageUrl={imageUrl}
        setImageUrl={setImageUrl}
        onSearch={handleSearch}
      />

      {hero && (
        <div className="description">
          <h2>{hero}</h2>
        </div>
      )}

      <div className="grid">
        {recommendations.length > 0 ? (
          recommendations.map((item, i) => (
            <ProductCard key={i} product={item} />
          ))
        ) : (
          <p>No recommendations yet. Try searching “chai” or “sofa”.</p>
        )}
      </div>
    </div>
  );
}

export default App;
