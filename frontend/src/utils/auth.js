export const setAuthState = (data) => {
  if (data.com_id) {
    localStorage.setItem("com_id", data.com_id);
  }
  if (data.session) {
    localStorage.setItem("session", data.session);
  }
};

export const getAuthState = () => {
  return {
    com_id: localStorage.getItem("com_id"),
    session: localStorage.getItem("session"),
  };
};

export const clearAuthState = () => {
  localStorage.removeItem("com_id");
  localStorage.removeItem("session");
};
