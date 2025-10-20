// frontend/src/pages/DataAnalysis.jsx
import React, { useEffect, useState } from "react";
import axios from "../api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell
} from "recharts";

export default function DataAnalysis() {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get("/analysis")
      .then(res => setData(res.data))
      .catch(err => console.error("Error fetching analysis:", err));
  }, []);

  if (!data) return <p>Loading analysis...</p>;

  const categoryData = Object.entries(data.category_counts || {}).map(([name, value]) => ({ name, value }));
  const keywordData = Object.entries(data.top_keywords || {}).map(([name, value]) => ({ name, value }));

  const COLORS = ["#2563eb", "#16a34a", "#f59e0b", "#ef4444", "#8b5cf6"];

  return (
    <div className="container">
      <h1>📊 Furniture Data Analysis Dashboard</h1>

      <div className="card">
        <h2>Total Products</h2>
        <p>{data.total_products}</p>
      </div>

      <div className="card">
        <h2>🪑 Category Distribution</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={categoryData}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#2563eb" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>💬 Top Keywords in Descriptions</h2>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={keywordData}
              dataKey="value"
              nameKey="name"
              outerRadius={120}
              label
            >
              {keywordData.map((_, index) => (
                <Cell key={index} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>💰 Average Price</h2>
        <p>${data.avg_price?.toFixed(2)}</p>
      </div>
    </div>
  );
}
