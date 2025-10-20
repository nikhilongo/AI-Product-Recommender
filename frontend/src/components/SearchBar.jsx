// frontend/src/components/SearchBar.jsx
import React from "react";

export default function SearchBar({ query, setQuery, imageUrl, setImageUrl, onSearch }) {
  return (
    <div className="search-bar">
      <input
        type="text"
        placeholder="Search for furniture (e.g., chai corner, study desk)"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <input
        type="text"
        placeholder="Optional: image URL"
        value={imageUrl}
        onChange={(e) => setImageUrl(e.target.value)}
      />
      <button onClick={onSearch}>Search</button>
    </div>
  );
}
