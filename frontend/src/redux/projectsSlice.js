import { createSlice } from "@reduxjs/toolkit"

const initialState = []

const projectsSlice = createSlice({
  name: "projects",
  initialState,
  reducers: {
    createProject: (state, action) => {
      state.push({
        projectId: action.payload.projectId,
        columns: action.payload.columns,
        importantColumnNames: action.payload.importantColumnNames,
        kpiList: action.payload.kpiList,
        uploadedFileData: action.payload.uploadedFileData,
        selectedKpi: action.payload.selectedKpi,
        droppedColumns: action.payload.droppedColumns,
      })
    },
    updateProject: (state, action) => {
      const index = state.findIndex((project) => project.projectId === action.payload.projectId)
      if (index !== -1) {
        const { columns, importantColumnNames, kpiList, uploadedFileData, selectedKpi, droppedColumns } = action.payload
        if (columns !== undefined) state[index].columns = columns
        if (importantColumnNames !== undefined) state[index].importantColumnNames = importantColumnNames
        if (kpiList !== undefined) state[index].kpiList = kpiList
        if (uploadedFileData !== undefined) state[index].uploadedFileData = uploadedFileData
        if (selectedKpi !== undefined) state[index].selectedKpi = selectedKpi
        if (droppedColumns !== undefined) state[index].droppedColumns = droppedColumns
      }
    },
    clearProject: (state, action) => {
      const index = state.findIndex((project) => project.projectId === action.payload.projectId)
      if (index !== -1) {
        state.splice(index, 1)
      }
    },
  },
})

export const { createProject, updateProject, clearProject } = projectsSlice.actions

export default projectsSlice.reducer

