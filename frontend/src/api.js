import axios from "axios";
import { ACCESS_TOKEN } from "./constants";

/* this is an intercepter. what is an interceptor?
intercepts any request we will send and automatically add the correct header
here we are using axios for network request
everytime we send a request it checks if we have access token */

const apiUrl = "/choreo-apis/awbo/backend/rest-api-be2/v1.0";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ? import.meta.env.VITE_API_URL : apiUrl,
});

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem(ACCESS_TOKEN);
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export default api;