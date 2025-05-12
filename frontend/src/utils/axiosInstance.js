import axios from "axios";
const BASE_URL = import.meta.env.VITE_BASE_URL;
console.log(BASE_URL);
const axiosInstance = axios.create({
  baseURL: BASE_URL, // FastAPI default port
  headers: {
    "Content-Type": "application/json",
  },
});

// Add a request interceptor
axiosInstance.interceptors.request.use(
  (config) => {
    // Get token and com_id from localStorage or sessionStorage
    const token =
      localStorage.getItem("token") || sessionStorage.getItem("token");
    const com_id =
      localStorage.getItem("com_id") || sessionStorage.getItem("com_id");

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    if (com_id) {
      config.headers["X-COM-ID"] = com_id;
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor
axiosInstance.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Clear token and com_id and redirect to login
      localStorage.removeItem("token");
      localStorage.removeItem("com_id");
      sessionStorage.removeItem("token");
      sessionStorage.removeItem("com_id");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

export default axiosInstance;
