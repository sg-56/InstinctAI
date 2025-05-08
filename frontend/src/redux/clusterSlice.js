import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  clusterHistory: [],
  selectedIndex: 0,
  clusters: null,
};

const clusterSlice = createSlice({
  name: "cluster",
  initialState,
  reducers: {
    setClusterHistory: (state, action) => {
      state.clusterHistory = action.payload;
    },
    setSelectedIndex: (state, action) => {
      console.log("-------", action, state);
      state.selectedIndex = action.payload;
    },
    addClusterToHistory: (state, action) => {
      state.clusterHistory.push(action.payload);
    },
    clearClusterHistory: (state) => {
      state.clusterHistory = [];
      state.selectedIndex = 0;
    },
    setClusters: (state, action) => {
      state.clusters = action.payload;
    },
  },
});

export const {
  setClusterHistory,
  setSelectedIndex,
  addClusterToHistory,
  clearClusterHistory,
  setClusters,
} = clusterSlice.actions;
export default clusterSlice.reducer;
