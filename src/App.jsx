import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

// --- 1. LOGIC & DATA (UNCHANGED) ---
const fetchMockProducts = () => {
    return new Promise((resolve) => {
        const products = [];
        for (let i = 1; i <= 100; i++) {
            products.push({
                id: i,
                name: `Item-${1000 + i}`, // Tech-style naming
                currentInventory: Math.floor(Math.random() * 100),
                avgSalesPerWeek: Math.floor(Math.random() * 50) + 5,
                leadTimeDays: Math.floor(Math.random() * 7) + 1,
            });
        }
        setTimeout(() => resolve(products), 1200);
    });
};

export default function InventoryPredictor() {
    const [products, setProducts] = useState([]);
    const [predictions, setPredictions] = useState({});
    const [loading, setLoading] = useState(false);
    const [modelStatus, setModelStatus] = useState("Initializing...");
    const [accuracy, setAccuracy] = useState(0);

    const [activeTab, setActiveTab] = useState('Overview');

    const trainModel = async () => {
        setModelStatus("Training Network...");
        const trainInputs = [];
        const trainLabels = [];
        for (let i = 0; i < 500; i++) {
            const stock = Math.floor(Math.random() * 100);
            const sales = Math.floor(Math.random() * 50) + 5;
            const lead = Math.floor(Math.random() * 7) + 1;
            const shouldReorder = stock < (sales / 7) * lead ? 1 : 0;
            trainInputs.push([stock, sales, lead]);
            trainLabels.push([shouldReorder]);
        }
        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [3], units: 8, activation: "relu" }));
        model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
        model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

        const history = await model.fit(tf.tensor2d(trainInputs), tf.tensor2d(trainLabels), { epochs: 30 });
        // Capture final accuracy
        setAccuracy((history.history.acc[history.history.acc.length - 1] * 100).toFixed(1));

        setModelStatus("Model Trained");
        return model;
    };

    useEffect(() => {
        const initApp = async () => {
            setLoading(true);
            const fetchedProducts = await fetchMockProducts();
            setProducts(fetchedProducts);
            const model = await trainModel();
            const inputData = fetchedProducts.map((p) => [p.currentInventory, p.avgSalesPerWeek, p.leadTimeDays]);
            const results = model.predict(tf.tensor2d(inputData));
            const values = await results.data();
            const newPredictions = {};
            fetchedProducts.forEach((p, index) => {
                newPredictions[p.id] = values[index] > 0.5 ? "Reorder" : "Healthy";
            });
            setPredictions(newPredictions);
            setLoading(false);
            setModelStatus("Ready");
        };
        initApp();
    }, []);

    // --- 2. VERCEL-STYLE UI COMPONENTS ---

    const StatusDot = ({ status }) => {
        const color = status === "Ready" || status === "Healthy" ? "#50e3c2" : status === "Reorder" ? "#ff0080" : "#f5a623";
        return (
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', backgroundColor: color, marginRight: 8, boxShadow: `0 0 8px ${color}` }}></span>
        );
    };

    const Card = ({ children, title }) => (
        <div style={{ border: '1px solid #333', borderRadius: 8, backgroundColor: '#111', overflow: 'hidden', marginBottom: 20 }}>
            {title && <div style={{ padding: '12px 20px', borderBottom: '1px solid #333', fontSize: 14, fontWeight: 600, color: '#888' }}>{title}</div>}
            <div style={{ padding: 20 }}>{children}</div>
        </div>
    );

    const [user, setUser] = useState(null); // null = not logged in, { name: '...' } = logged in
    const [authView, setAuthView] = useState('login'); // 'login' or 'register'
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = (e) => {
        e.preventDefault();
        if (email && password) {
            setUser({ name: email.split('@')[0] });
        }
    };

    const handleRegister = (e) => {
        e.preventDefault();
        if (email && password) {
            setUser({ name: email.split('@')[0] });
        }
    };

    const handleLogout = () => {
        setUser(null);
        setEmail('');
        setPassword('');
        setAuthView('login');
    };

    if (!user) {
        return (
            <div style={{ minHeight: "100vh", backgroundColor: "#000", color: "#fff", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ width: '100%', maxWidth: 400, padding: 40, border: '1px solid #333', borderRadius: 8, backgroundColor: '#111' }}>
                    <div style={{ textAlign: 'center', marginBottom: 30 }}>
                        <div style={{ fontSize: 40, marginBottom: 10 }}>🧠</div>
                        <h1 style={{ fontSize: 24, fontWeight: 700 }}>{authView === 'login' ? 'Log In' : 'Sign Up'}</h1>
                        <p style={{ color: '#888', marginTop: 10 }}>Enter your credentials to access the dashboard.</p>
                    </div>

                    <form onSubmit={authView === 'login' ? handleLogin : handleRegister} style={{ display: 'flex', flexDirection: 'column', gap: 15 }}>
                        <div>
                            <label style={{ display: 'block', marginBottom: 8, fontSize: 14, color: '#888' }}>Email Address</label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="name@example.com"
                                style={{ width: '100%', padding: '10px 12px', background: '#000', border: '1px solid #333', borderRadius: 5, color: '#fff', fontSize: 14, boxSizing: 'border-box' }}
                                required
                            />
                        </div>
                        <div>
                            <label style={{ display: 'block', marginBottom: 8, fontSize: 14, color: '#888' }}>Password</label>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="••••••••"
                                style={{ width: '100%', padding: '10px 12px', background: '#000', border: '1px solid #333', borderRadius: 5, color: '#fff', fontSize: 14, boxSizing: 'border-box' }}
                                required
                            />
                        </div>
                        <button
                            type="submit"
                            style={{ marginTop: 10, padding: '12px', background: '#fff', color: '#000', border: 'none', borderRadius: 5, fontWeight: 600, cursor: 'pointer', fontSize: 14 }}
                        >
                            {authView === 'login' ? 'Continue with Email' : 'Create Account'}
                        </button>
                    </form>

                    <div style={{ marginTop: 20, textAlign: 'center', fontSize: 14, color: '#888' }}>
                        {authView === 'login' ? "Don't have an account? " : "Already have an account? "}
                        <span
                            onClick={() => setAuthView(authView === 'login' ? 'register' : 'login')}
                            style={{ color: '#fff', cursor: 'pointer', textDecoration: 'underline' }}
                        >
                            {authView === 'login' ? 'Sign Up' : 'Log In'}
                        </span>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div style={{ minHeight: "100vh", backgroundColor: "#000", color: "#fff", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" }}>

            {/* HEADER */}
            <header style={{ borderBottom: '1px solid #333', padding: '0 24px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between', backgroundColor: '#0a0a0a' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{ width: 24, height: 24, background: '#fff', borderRadius: '50%' }}></div>
                    <span onClick={() => window.location.reload()} style={{ fontWeight: 600, cursor: 'pointer' }}>{user.name}</span>
                    <span style={{ color: '#444' }}>/</span>
                    <span onClick={() => window.location.reload()} style={{ fontWeight: 600, cursor: 'pointer' }}>inventory-ai-forecast</span>
                </div>
                <div style={{ display: 'flex', gap: 10 }}>
                    <button onClick={handleLogout} style={{ background: 'transparent', border: '1px solid #333', color: '#fff', padding: '6px 12px', borderRadius: 5, fontSize: 13, cursor: 'pointer' }}>Log Out</button>
                    <button onClick={() => alert("Feedback form opening...")} style={{ background: 'transparent', border: '1px solid #333', color: '#fff', padding: '6px 12px', borderRadius: 5, fontSize: 13, cursor: 'pointer' }}>Feedback</button>
                    <button onClick={() => window.open('https://vercel.com', '_blank')} style={{ background: '#fff', border: 'none', color: '#000', padding: '6px 12px', borderRadius: 5, fontSize: 13, fontWeight: 600, cursor: 'pointer' }}>Visit</button>
                </div>
            </header>

            {/* NAV TABS */}
            <div style={{ borderBottom: '1px solid #333', padding: '0 24px', display: 'flex', gap: 24, fontSize: 14, color: '#888' }}>
                {['Overview', 'Deployments', 'Analytics', 'Logs'].map((tab) => (
                    <div
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        style={{
                            padding: '12px 0',
                            borderBottom: activeTab === tab ? '2px solid #fff' : '2px solid transparent',
                            color: activeTab === tab ? '#fff' : '#888',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease'
                        }}
                    >
                        {tab}
                    </div>
                ))}
            </div>

            {/* MAIN CONTENT */}
            <main style={{ maxWidth: 1000, margin: '40px auto', padding: '0 20px' }}>

                {activeTab === 'Overview' && (
                    <>
                        <h1 style={{ fontSize: 32, fontWeight: 700, marginBottom: 30 }}>Production Deployment</h1>
                        {/* DEPLOYMENT CARD */}
                        <div style={{ display: 'flex', flexWrap: 'wrap', border: '1px solid #333', borderRadius: 8, backgroundColor: '#000', marginBottom: 40 }}>
                            {/* Left Preview Area */}
                            <div style={{ flex: '1 1 300px', borderRight: '1px solid #333', padding: 40, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(45deg, #111, #050505)', minHeight: '200px' }}>
                                <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontSize: 60, marginBottom: 10 }}>🧠</div>
                                    <div style={{ fontSize: 24, fontWeight: 700, color: '#fff' }}>TensorFlow Model</div>
                                    <div style={{ color: '#666', marginTop: 10 }}>Architecture: Dense(8) -&gt; Dense(1)</div>
                                </div>
                            </div>

                            {/* Right Details Area */}
                            <div style={{ flex: '1 1 300px', padding: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
                                <div>
                                    <div style={{ color: '#666', fontSize: 12, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>Status</div>
                                    <div style={{ display: 'flex', alignItems: 'center', fontWeight: 500 }}>
                                        <StatusDot status={loading ? "Initializing" : "Ready"} />
                                        {loading ? modelStatus : "Ready"}
                                    </div>
                                </div>
                                <div>
                                    <div style={{ color: '#666', fontSize: 12, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>Model Accuracy</div>
                                    <div style={{ fontWeight: 500 }}>{loading ? "Calculating..." : `${accuracy}%`}</div>
                                </div>
                                <div>
                                    <div style={{ color: '#666', fontSize: 12, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>Source</div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                        <span style={{ fontFamily: 'monospace', background: '#333', padding: '2px 6px', borderRadius: 4, fontSize: 12 }}>src/App.js</span>
                                    </div>
                                </div>
                                <div style={{ marginTop: 'auto', paddingTop: 20 }}>
                                    <button
                                        onClick={() => setActiveTab('Logs')}
                                        style={{ width: '100%', padding: '10px', background: '#fff', border: 'none', borderRadius: 4, fontWeight: 600, cursor: 'pointer' }}
                                    >
                                        View Logs
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* DATA TABLE SECTION */}
                        <h2 style={{ fontSize: 20, marginBottom: 15 }}>Inventory Analysis</h2>
                        <Card>
                            {loading ? (
                                <div style={{ textAlign: 'center', padding: 40, color: '#666' }}>Initializing AI Engine...</div>
                            ) : (
                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
                                    <thead>
                                        <tr style={{ color: '#666', textAlign: 'left' }}>
                                            <th style={{ padding: '10px 0', borderBottom: '1px solid #333' }}>Product Name</th>
                                            <th style={{ padding: '10px 0', borderBottom: '1px solid #333' }}>Stock Level</th>
                                            <th style={{ padding: '10px 0', borderBottom: '1px solid #333' }}>Demand (Wk)</th>
                                            <th style={{ padding: '10px 0', borderBottom: '1px solid #333' }}>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {products.map((p) => {
                                            const status = predictions[p.id];
                                            const isBad = status === "Reorder";
                                            return (
                                                <tr key={p.id} style={{ borderBottom: '1px solid #222' }}>
                                                    <td style={{ padding: '14px 0', fontWeight: 500 }}>{p.name}</td>
                                                    <td style={{ padding: '14px 0', color: '#888' }}>{p.currentInventory} units</td>
                                                    <td style={{ padding: '14px 0', color: '#888' }}>{p.avgSalesPerWeek}</td>
                                                    <td style={{ padding: '14px 0' }}>
                                                        <span style={{
                                                            background: isBad ? 'rgba(255,0,128,0.2)' : 'rgba(80,227,194,0.1)',
                                                            color: isBad ? '#ff0080' : '#50e3c2',
                                                            padding: '4px 10px',
                                                            borderRadius: 20,
                                                            fontSize: 12,
                                                            fontWeight: 500,
                                                            border: `1px solid ${isBad ? '#ff0080' : '#50e3c2'}`
                                                        }}>
                                                            {status}
                                                        </span>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            )}
                        </Card>
                    </>
                )}

                {activeTab === 'Deployments' && (
                    <>
                        <h1 style={{ fontSize: 32, fontWeight: 700, marginBottom: 30 }}>Deployment History</h1>
                        <Card>
                            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
                                <thead>
                                    <tr style={{ color: '#666', textAlign: 'left' }}>
                                        <th style={{ padding: '10px 0', borderBottom: '1px solid #333' }}>Deployment ID</th>
                                        <th style={{ padding: '10px 0', borderBottom: '1px solid #333' }}>Status</th>
                                        <th style={{ padding: '10px 0', borderBottom: '1px solid #333' }}>Environment</th>
                                        <th style={{ padding: '10px 0', borderBottom: '1px solid #333' }}>Time</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {[
                                        { id: 'dpl_8f7e2a', status: 'Ready', env: 'Production', time: 'Just now' },
                                        { id: 'dpl_3c9b1x', status: 'Ready', env: 'Preview', time: '2h ago' },
                                        { id: 'dpl_7v8n2m', status: 'Error', env: 'Preview', time: '5h ago' },
                                        { id: 'dpl_1q2w3e', status: 'Ready', env: 'Production', time: '1d ago' },
                                    ].map((d) => (
                                        <tr key={d.id} style={{ borderBottom: '1px solid #222' }}>
                                            <td style={{ padding: '14px 0', fontFamily: 'monospace' }}>{d.id}</td>
                                            <td style={{ padding: '14px 0' }}>
                                                <StatusDot status={d.status === 'Ready' ? 'Ready' : 'Reorder'} />
                                                {d.status}
                                            </td>
                                            <td style={{ padding: '14px 0' }}>{d.env}</td>
                                            <td style={{ padding: '14px 0', color: '#888' }}>{d.time}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </Card>
                    </>
                )}

                {activeTab === 'Analytics' && (
                    <>
                        <h1 style={{ fontSize: 32, fontWeight: 700, marginBottom: 30 }}>Analytics</h1>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 20, marginBottom: 40 }}>
                            <Card title="Total Products">
                                <div style={{ fontSize: 36, fontWeight: 700 }}>{products.length}</div>
                            </Card>
                            <Card title="Reorder Rate">
                                <div style={{ fontSize: 36, fontWeight: 700 }}>
                                    {products.length > 0
                                        ? Math.round((Object.values(predictions).filter(s => s === 'Reorder').length / products.length) * 100)
                                        : 0}%
                                </div>
                            </Card>
                            <Card title="Model Accuracy">
                                <div style={{ fontSize: 36, fontWeight: 700 }}>{accuracy}%</div>
                            </Card>
                        </div>
                        <Card title="Inventory Distribution">
                            <div style={{ display: 'flex', alignItems: 'flex-end', height: 200, gap: 4, padding: '20px 0' }}>
                                {products.slice(0, 50).map((p, i) => (
                                    <div
                                        key={i}
                                        style={{
                                            flex: 1,
                                            background: predictions[p.id] === 'Reorder' ? '#ff0080' : '#50e3c2',
                                            height: `${p.currentInventory}%`,
                                            opacity: 0.7,
                                            borderRadius: '2px 2px 0 0'
                                        }}
                                        title={`${p.name}: ${p.currentInventory}`}
                                    />
                                ))}
                            </div>
                            <div style={{ textAlign: 'center', color: '#666', fontSize: 12, marginTop: 10 }}>Sample of 50 Items Inventory Levels</div>
                        </Card>
                    </>
                )}

                {activeTab === 'Logs' && (
                    <>
                        <h1 style={{ fontSize: 32, fontWeight: 700, marginBottom: 30 }}>System Logs</h1>
                        <Card>
                            <div style={{ fontFamily: 'monospace', fontSize: 13, color: '#888', lineHeight: 1.6 }}>
                                <div>[system] Initializing TensorFlow.js backend...</div>
                                <div>[system] Backend 'webgl' configured.</div>
                                <div>[model] Loading sequential model architecture...</div>
                                <div>[model] Compiling with optimizer 'adam'...</div>
                                {loading ? (
                                    <>
                                        <div style={{ color: '#f5a623' }}>[training] {modelStatus}</div>
                                        <div>[training] Epoch 1/30 - loss: 0.6932 - acc: 0.5000</div>
                                        <div>[training] Epoch 5/30 - loss: 0.5421 - acc: 0.7200</div>
                                        <div>[training] Epoch 10/30 - loss: 0.3102 - acc: 0.8900</div>
                                        <div>[training] Processing...</div>
                                    </>
                                ) : (
                                    <>
                                        <div>[training] Model trained successfully. Final Accuracy: {accuracy}%</div>
                                        <div style={{ color: '#50e3c2' }}>[success] Prediction engine ready.</div>
                                        <div>[api] Fetched {products.length} products from mock database.</div>
                                        <div>[inference] Processed {products.length} items in 45ms.</div>
                                        <div>[system] Listening for updates...</div>
                                    </>
                                )}
                            </div>
                        </Card>
                    </>
                )}

            </main>
        </div>
    );
}
