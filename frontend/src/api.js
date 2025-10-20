import axios from "axios";

export default axios.create({
  baseURL: "https://furniture-ai.onrender.com", // ✅ your Render backend URL
});
