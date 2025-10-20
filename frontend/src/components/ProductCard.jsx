// frontend/src/components/ProductCard.jsx
import React from "react";

export default function ProductCard({ product }) {
  return (
    <div className="product-card">
      <h3>{product.title}</h3>
      <p className="short">{product.short}</p>
      <ul>
        {product.features?.map((f, i) => (
          <li key={i}>{f}</li>
        ))}
      </ul>
      <button className="cta">{product.cta || "Learn More"}</button>
    </div>
  );
}
