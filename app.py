import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import tensorflow as tf

# 1. Cấu hình trang Streamlit (Wide Layout)
st.set_page_config(
    page_title="AI Stock Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cố định Seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# CSS Tùy chỉnh Giao diện
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2a2e39;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Thanh điều hướng Sidebar
st.sidebar.title("⚙️ Cấu Hình Dự Báo")
ticker_input = st.sidebar.text_input("Mã chứng khoán (Yahoo Finance):", value="VNM.VN")
start_date = st.sidebar.date_input("Ngày bắt đầu:", pd.to_datetime("2018-01-01"))
forecast_days = st.sidebar.slider("Số ngày dự báo (T+N):", min_value=1, max_value=10, value=7)

st.sidebar.subheader("Tham số Mô hình LSTM")
epochs = st.sidebar.slider("Số chu kỳ (Epochs):", min_value=5, max_value=30, value=10)
batch_size = st.sidebar.select_slider("Batch Size:", options=[16, 32, 64], value=32)

# 3. Tải Dữ liệu
@st.cache_data(ttl=3600)
def load_stock_data(ticker, start):
    try:
        df = yf.download(ticker, start=start)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        return None

df = load_stock_data(ticker_input, start_date)

if df is None or df.empty:
    st.error(f"❌ Không tìm thấy dữ liệu cho mã '{ticker_input}'. Vui lòng kiểm tra lại mã chứng khoán (ví dụ: VNM.VN, FPT.VN, AAPL).")
    st.stop()

# Đảm bảo các cột cần thiết tồn tại
if 'Close' not in df.columns:
    st.error("Dữ liệu tải về không hợp lệ.")
    st.stop()

# 4. Tiền xử lý dữ liệu & Huấn luyện
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, y_train = create_dataset(train_data, look_back)

# Chuẩn bị dữ liệu Test
test_inputs = np.vstack((train_data[-look_back:], test_data))
X_test, y_test = create_dataset(test_inputs, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

@st.cache_resource
def train_lstm_model(X_tr, y_tr, epochs_cnt, b_size):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_tr.shape[1], 1)),
        Dropout(0.1),
        LSTM(50, return_sequences=False),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_tr, y_tr, epochs=epochs_cnt, batch_size=b_size, verbose=0)
    return model

with st.spinner("🔄 Đang huấn luyện mô hình Deep Learning (LSTM)..."):
    model = train_lstm_model(X_train, y_train, epochs, batch_size)

# Dự báo dữ liệu test
test_preds = model.predict(X_test)
test_preds = scaler.inverse_transform(test_preds)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Dự báo tương lai
last_sequence = scaled_data[-look_back:]
future_preds = []
curr_seq = last_sequence.reshape(1, look_back, 1)

for _ in range(forecast_days):
    pred = model.predict(curr_seq, verbose=0)
    future_preds.append(pred[0, 0])
    curr_seq = np.append(curr_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

# Tính toán các chỉ số thống kê
latest_price = float(df['Close'].iloc[-1])
target_price = float(future_preds[-1])
pnl_pct = ((target_price - latest_price) / latest_price) * 100
pnl_amount = target_price - latest_price

# 5. Header & Metric Overview
st.title(f"📈 Dashboard Phân Tích & Dự Báo: {ticker_input.upper()}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Giá Hiện Tại (Gần Nhất)", f"{latest_price:,.2f}")
col2.metric(f"Dự Báo Ngày T+{forecast_days}", f"{target_price:,.2f}", delta=f"{pnl_amount:+,.2f} ({pnl_pct:+.2f}%)")
col3.metric("Giá Cao Nhất Dự Báo", f"{np.max(future_preds):,.2f}")
col4.metric("Giá Thấp Nhất Dự Báo", f"{np.min(future_preds):,.2f}")

st.markdown("---")

# 6. Các Tab Chức Năng
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Phân Tích Kỹ Thuật", 
    "🔮 Dự Báo Tương Lai", 
    "🎯 Đánh Giá Mô Hình", 
    "📋 Dữ Liệu Bảng"
])

# TAB 1: PHÂN TÍCH KỸ THUẬT & NẾN
with tab1:
    st.subheader("Biểu Đồ Nến & Chỉ Báo Kỹ Thuật (SMA 20/50)")
    
    # Tính SMA
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    fig_candlestick = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick
    fig_candlestick.add_trace(go.Candlestick(
        x=df.index,
open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Giá Nến"
    ), row=1, col=1)
    
    # SMA 20 & 50
    fig_candlestick.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name="SMA 20", line=dict(color='orange', width=1)), row=1, col=1)
    fig_candlestick.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA 50", line=dict(color='blue', width=1)), row=1, col=1)
    
    # Khối lượng Volume
    if 'Volume' in df.columns:
        fig_candlestick.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Khối Lượng", marker_color='gray'), row=2, col=1)
        
    fig_candlestick.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_candlestick, use_container_width=True)

# TAB 2: DỰ BÁO TƯƠNG LAI
with tab2:
    st.subheader(f"Xu Hướng Dự Báo Trong {forecast_days} Ngày Tới")
    
    # Tạo ngày dự báo tương lai
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    
    # Biểu đồ Plotly tương tác
    fig_forecast = go.Figure()
    
    # Lấy 90 ngày quá khứ
    recent_df = df.tail(90)
    fig_forecast.add_trace(go.Scatter(
        x=recent_df.index, y=recent_df['Close'],
        mode='lines+markers', name='Giá Thực Tế (90 Ngày Gần Nhất)',
        line=dict(color='#00F2FE', width=2)
    ))
    
    # Đường nối từ giá thực tế cuối cùng sang ngày dự báo đầu tiên
    forecast_x = [recent_df.index[-1]] + list(future_dates)
    forecast_y = [recent_df['Close'].iloc[-1]] + list(future_preds)
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y,
        mode='lines+markers', name='Dự Báo Tương Lai',
        line=dict(color='#FF0844', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig_forecast.update_layout(
        height=500,
        template="plotly_dark",
        xaxis_title="Thời Gian",
        yaxis_title="Giá (VND/USD)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Tín hiệu đề xuất
    st.markdown("### 🎯 Nhận Định Xu Hướng")
    if pnl_pct > 2.0:
        st.success(f"🟢 **Tín hiệu Khả quan:** Mô hình dự báo giá có xu hướng TĂNG mạnh khoảng **{pnl_pct:+.2f}%** trong {forecast_days} ngày tới.")
    elif pnl_pct < -2.0:
        st.error(f"🔴 **Tín hiệu Cảnh báo:** Mô hình dự báo giá có xu hướng GIẢM khoảng **{pnl_pct:+.2f}%** trong {forecast_days} ngày tới.")
    else:
        st.warning(f"🟡 **Tín hiệu Đi Ngang:** Biến động giá không quá đáng kể (**{pnl_pct:+.2f}%**). Có thể tiếp tục theo dõi.")

# TAB 3: ĐÁNH GIÁ MÔ HÌNH
with tab3:
    st.subheader("Đánh Giá Độ Chính Xác Mô Hình Trên Tập Test")
rmse = np.sqrt(mean_squared_error(y_test_actual, test_preds))
    mae = mean_absolute_error(y_test_actual, test_preds)
    r2 = r2_score(y_test_actual, test_preds)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("RMSE (Root Mean Sq Error)", f"{rmse:.4f}")
    col_m2.metric("MAE (Mean Absolute Error)", f"{mae:.4f}")
    col_m3.metric("R² Score (Độ Phù Hợp)", f"{r2:.4f}")
    
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(y=y_test_actual.flatten(), mode='lines', name='Giá Thực Tế (Test)', line=dict(color='white')))
    fig_eval.add_trace(go.Scatter(y=test_preds.flatten(), mode='lines', name='Giá Mô Hình Dự Báo', line=dict(color='#00FF88', width=1.5)))
    fig_eval.update_layout(height=450, template="plotly_dark", title="So Sánh Giá Thực Tế vs Dự Báo Trên Tập Kiểm Thử")
    st.plotly_chart(fig_eval, use_container_width=True)

# TAB 4: BẢNG DỮ LIỆU
with tab4:
    st.subheader("Chi Tiết Giá Dự Báo Theo Ngày")
    
    forecast_table = pd.DataFrame({
        'Ngày Dự Báo': future_dates.strftime('%Y-%m-%d'),
        'Giá Dự Báo': np.round(future_preds, 2),
        'Chênh Lệch So Với Hiện Tại': np.round(future_preds - latest_price, 2),
        'Lỗ/Lãi Dự Kiến (%)': np.round(((future_preds - latest_price) / latest_price) * 100, 2)
    })
    
    st.dataframe(forecast_table.style.highlight_max(axis=0, color='#1f77b4'), use_container_width=True)
