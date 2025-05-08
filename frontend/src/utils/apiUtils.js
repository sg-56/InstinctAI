import axiosInstance from "./axiosInstance";

const baseUrl = import.meta.env.VITE_BASE_URL;

export const featureRanking = async (
  project_id,
  kpi,
  importantColumnNames,
  kpiList
) => {
  try {
    const response = await axiosInstance.post(
      `/projects/${project_id}/features/ranking`,
      {
        project_id,
        kpi: kpi,
        important_features: importantColumnNames,
        kpi_list: kpiList,
      }
    );
    console.log("Fetching feature ranking:", response);
    return response.data.task_id;
  } catch (error) {
    console.error("Error:", error);
    throw error;
  }
};

export const timeSeriesApi = {
  fetchColumns: async (baseUrl, project_id) => {
    const response = await axiosInstance.post(
      `/projects/${project_id}/dataset/columns`,
      { project_id }
    );
    return response.data.columns;
  },

  runAnalysis: async (baseUrl, params) => {
    const response = await axiosInstance.post(
      `/projects/${params.project_id}/time-series/analysis`,
      params
    );
    return response.data;
  },

  checkTaskStatus: async (baseUrl, task_id) => {
    const response = await axiosInstance.get(
      `/projects/tasks/${task_id}/status`
    );
    return response.data;
  },

  fetchTimeSeriesFigure: async (baseUrl, params) => {
    const response = await axiosInstance.post(
      `/projects/${params.project_id}/time-series/figure`,
      params
    );
    return response.data;
  },
};
