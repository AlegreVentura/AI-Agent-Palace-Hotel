document.addEventListener('DOMContentLoaded', () => {
    const allHotels = [
        { id: "beach_palace", name: "Beach Palace", city: "Cancún", country: "Mexico", brand: "Palace Resorts", img: "beach-palace-cancun.jpg" },
        { id: "cozumel_palace", name: "Cozumel Palace", city: "Cozumel", country: "Mexico", brand: "Palace Resorts", img: "cozumel-palace.jpeg" },
        { id: "playacar_palace", name: "Playacar Palace", city: "Playa del Carmen", country: "Mexico", brand: "Palace Resorts", img: "playacar-palace.jpg" },
        { id: "moon_cancun", name: "Moon Palace Cancún", city: "Cancún", country: "Mexico", brand: "Moon Palace Resorts", img: "moon-palace-cancun.jpg" },
        { id: "leblanc_cancun", name: "Le Blanc Spa Resort Cancún", city: "Cancún", country: "Mexico", brand: "Le Blanc Spa Resorts", img: "le-blanc-spa-resort-cancun.jpg" },
        { id: "leblanc_cabo", name: "Le Blanc Spa Resort Los Cabos", city: "Los Cabos", country: "Mexico", brand: "Le Blanc Spa Resorts", img: "le-blanc-spa-resort-los-cabos.jpg" },
        { id: "baglioni_maldives", name: "Baglioni Resort Maldives", city: "Maagau", country: "Maldives", brand: "Baglioni Hotels", img: "Baglioni Resort Maldives.jpg" },
        { id: "baglioni_sardinia", name: "Baglioni Resort Sardinia", city: "Sardinia", country: "Italy", brand: "Baglioni Hotels", img: "Baglioni Resort Sardinia.jpg" },
    ];
    const clientListContainer = document.getElementById('client-list');
    let hotelDataStore = {};
    let currentClientData = {};
    let scenarioQueue = ['familia', 'romantico', 'live'];
    let clientDataStore = {};
    let currentScenario = null;
    let ws = null;

    function renderClientList() {
        clientListContainer.querySelectorAll('.client-item').forEach(item => item.remove());
        const clientTitle = document.createElement('li');
        clientTitle.className = 'nav-item mt-3';
        clientTitle.innerHTML = `<h6 class="ps-4 ms-2 text-uppercase text-xs text-dark font-weight-bolder opacity-8">Clients</h6>`;
        clientListContainer.appendChild(clientTitle);

        scenarioQueue.forEach(scenario => {
            const clientItem = document.createElement('li');
            clientItem.className = 'nav-item client-item';
            clientItem.dataset.scenario = scenario;
            let clientName = '';
            switch (scenario) {
                case 'familia':
                    clientName = 'Moon Family';
                    break;
                case 'romantico':
                    clientName = 'Le Blanc Couple';
                    break;
                case 'live':
                    clientName = 'Live Data';
                    break;
            }
            clientItem.innerHTML = `<a class="nav-link" href="#"><span class="nav-link-text ms-1">${clientName}</span></a>`;
            clientItem.addEventListener('click', (e) => {
                e.preventDefault();
                loadScenario(scenario);
            });
            clientListContainer.appendChild(clientItem);
        });
        updateClientListActiveState();
    }

    function updateClientListActiveState() {
        document.querySelectorAll('.client-item .nav-link').forEach(link => link.classList.remove('active'));
        const activeItem = document.querySelector(`.client-item[data-scenario="${currentScenario}"] .nav-link`);
        if (activeItem) {
            activeItem.classList.add('active');
        }
    }

    async function fetchStaticData(scenario) {
        const response = await fetch(`datos/client_${scenario}_actualizado.json?t=${new Date().getTime()}`);
        return await response.json();
    }

    async function loadScenario(scenario) {
        currentScenario = scenario;
        updateClientListActiveState();

        if (scenario === 'live') {
            if (!ws || ws.readyState === WebSocket.CLOSED) {
                connectWebSocket();
            }
        } else {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
            if (!clientDataStore[scenario]) {
                clientDataStore[scenario] = await fetchStaticData(scenario);
            }
            const data = clientDataStore[scenario];
            updateUI(data);
        }
    }

    function connectWebSocket() {
        ws = new WebSocket(`ws://localhost:3033`);

        ws.onopen = () => console.log('WebSocket connection established.');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (currentScenario === 'live') {
                currentClientData = data;
                updateUI(data);
            }
        };
        ws.onclose = () => {
            console.log('WebSocket connection closed.');
        };
        ws.onerror = (error) => console.error('WebSocket error:', error);
    }

    async function fetchHotelData() {
        const response = await fetch(`datos/hoteles_palace_dataset.json?t=${new Date().getTime()}`);
        return await response.json();
    }

    function updateUI(data) {
        if (!data) {
            showNoClientsView();
            return;
        }
        document.getElementById('main-dashboard-content').style.display = 'block';
        document.getElementById('no-clients-view').style.display = 'none';
        
        displaySummary(data);
        displayClientInfo(data.client);
        displayPreferences(data.client, data.apis);
        displayRecommendedHotels(data.client.resort);
        displayCallStatus(data.llamada);
        displayEmotion(data.analisis_emociones);
        if (data.apis?.availability?.request) {
            initializeCalendar(data.apis.availability.request.check_in_date, data.apis.availability.request.check_out_date);
        }
        displayCallSummary(data.llamada, data.resumen_llamada_md);
    }

    function displaySummary(data) {
        const container = document.getElementById('summary-cards');
        container.innerHTML = '';
        if (!data || !data.client) return;
        const client = data.client;
        const pricing = data.apis?.pricing?.response;
        const summaryData = [
            { title: 'Guest Type', key: 'tipo_huesped', value: (client.tipo_huesped || '').replace(/_/g, ' '), icon: 'person', isPrice: false },
            { title: 'Resort', key: 'resort', value: (client.resort || '').replace(/_/g, ' '), icon: 'hotel', isPrice: false },
            { title: 'Total Price (USD)', key: 'total_price_usd', value: pricing?.summary?.total_price_usd || '0', icon: 'price_check', isPrice: true }
        ];
        summaryData.forEach(item => {
            const col = document.createElement('div');
            col.className = 'col-xl-4 col-sm-6 mb-xl-0 mb-4';
            let valueElement = item.isPrice
                ? `<div class="price-input-group"><span class="currency-symbol">$</span><input type="number" step="0.01" class="form-control" data-key="${item.key}" data-price="true" value="${item.value}"></div>`
                : `<input type="text" class="form-control" data-key="${item.key}" value="${item.value}">`;
            col.innerHTML = `
                <div class="card"><div class="card-header p-2 ps-3"><div class="d-flex align-items-center">
                    <div class="flex-grow-1"><p class="text-sm mb-0 text-capitalize">${item.title}</p>${valueElement}</div>
                    <div class="icon icon-md icon-shape bg-gradient-dark shadow-dark shadow text-center border-radius-lg ms-3"><i class="material-symbols-rounded opacity-10">${item.icon}</i></div>
                </div></div></div>`;
            container.appendChild(col);
        });
    }

    function displayClientInfo(client) {
        const container = document.getElementById('personal-info');
        container.innerHTML = '<h3 class="card-title">Personal Data</h3>';
        const map = { "Name": "nombre", "Email": "email", "Phone": "telefono", "Member Status": "member_status" };
        for (const [label, key] of Object.entries(map)) {
            const value = key === 'Name' ? `${client.nombre} ${client.apellidos}` : client[key];
            const formGroup = document.createElement('div');
            formGroup.className = 'form-group';
            formGroup.innerHTML = `<label>${label}</label><input type="text" class="form-control" value="${value}">`;
            container.appendChild(formGroup);
        }
    }

    function displayPreferences(client, apis) {
        const container = document.getElementById('preferences-info');
        container.innerHTML = `<h3 class="card-title">Preferences</h3>`;
        container.innerHTML += `<div class="form-group"><label>Country</label><select class="form-control" id="country-select"></select></div>`;
        container.innerHTML += `<div class="form-group"><label>Resort</label><select class="form-control" id="resort-select"></select></div>`;
        container.innerHTML += `<div class="form-group"><label>Room Type</label><input type="text" class="form-control" value="${apis.availability.response.room_type.replace(/_/g, ' ')}"></div>`;
        
        populateCountrySelector(client.pais);
        populateResortSelector(client.pais, client.resort);

        const countrySelect = document.getElementById('country-select');
        const resortSelect = document.getElementById('resort-select');

        countrySelect.addEventListener('change', () => {
            const selectedCountry = countrySelect.value;
            populateResortSelector(selectedCountry);
            const firstResort = resortSelect.value;
            if (firstResort) {
                displayRecommendedHotels(firstResort);
            } else {
                // If there are no resorts for the selected country, clear the hotel card
                document.getElementById('hotel-cards').innerHTML = '';
            }
        });

        resortSelect.addEventListener('change', () => {
            const selectedResort = resortSelect.value;
            displayRecommendedHotels(selectedResort);
        });
    }

    function populateCountrySelector(selectedCountry) {
        const select = document.getElementById('country-select');
        select.innerHTML = '';
        for (const countryCode in hotelDataStore.por_pais) {
            const option = document.createElement('option');
            option.value = countryCode;
            option.textContent = hotelDataStore.por_pais[countryCode].pais;
            if (countryCode === selectedCountry) option.selected = true;
            select.appendChild(option);
        }
    }

    function populateResortSelector(countryCode, selectedResort) {
        const select = document.getElementById('resort-select');
        select.innerHTML = '';
        const countryData = hotelDataStore.por_pais[countryCode];
        if (countryData) {
            countryData.hoteles.forEach(hotel => {
                const option = document.createElement('option');
                option.value = hotel.id;
                option.textContent = hotel.nombre;
                if (hotel.id === selectedResort) option.selected = true;
                select.appendChild(option);
            });
        }
    }

    function createHotelCard(hotel) {
        const col = document.createElement('div');
        col.className = 'col-12 mb-4';
        const imgSrc = hotel.img ? `fotos/${hotel.img}` : '';
        col.innerHTML = `<div class="card">${imgSrc ? `<img src="${imgSrc}" class="card-img-top" alt="Image of ${hotel.name}">` : ''}<div class="card-body"><h5 class="card-title">${hotel.name}</h5><p class="card-text">${hotel.city}, ${hotel.country}</p></div></div>`;
        return col;
    }

    function displayRecommendedHotels(hotelIds) {
        const container = document.getElementById('hotel-cards');
        container.innerHTML = '';
        (Array.isArray(hotelIds) ? hotelIds : [hotelIds]).forEach(hotelId => {
            const hotelInfo = allHotels.find(h => h.id === hotelId);
            if (hotelInfo) container.appendChild(createHotelCard(hotelInfo));
        });
    }

    function showNoClientsView() {
        document.getElementById('main-dashboard-content').style.display = 'none';
        document.getElementById('no-clients-view').style.display = 'block';
    }

    function displayCallStatus(llamada) {
        const banner = document.getElementById('call-status-banner');
        if (llamada) {
            banner.textContent = llamada.ongoing_call === 1 ? 'Ongoing Call' : 'No Active Call';
            banner.className = `alert ${llamada.ongoing_call === 1 ? 'ongoing-call' : 'no-call'}`;
        } else {
            banner.textContent = 'No Active Call';
            banner.className = 'alert no-call';
        }
    }

    function displayCallSummary(llamada, markdown) {
        const summaryCard = document.getElementById('call-summary-card');
        const summaryContent = document.getElementById('call-summary-content');
        if (llamada?.ongoing_call === 0 && markdown) {
            summaryCard.style.display = 'block';
            summaryContent.innerHTML = markdown.replace(/### (.*)/g, '').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>').replace(/> (.*)/g, '<blockquote>$1</blockquote>').replace(/\n/g, '<br>');
        } else {
            summaryCard.style.display = 'none';
        }
    }

    function displayEmotion(analisis) {
        const emotionIndicator = document.getElementById('emotion-indicator');
        const emotions = ['😠', '😐', '😊'];
        if (analisis?.emocion_principal !== undefined) {
            emotionIndicator.textContent = emotions[analisis.emocion_principal];
        }
    }

    function initializeCalendar(checkIn, checkOut) {
        flatpickr("#date-picker", { mode: "range", dateFormat: "Y-m-d", defaultDate: [checkIn, checkOut] });
    }

    function showConfirmationModal() {
        const data = currentClientData;
        if (!data || !data.client) return;
        const clientName = `${data.client.nombre} ${data.client.apellidos}`;
        const resortName = (data.client.resort || '').replace(/_/g, ' ');
        const dates = document.getElementById('date-picker').value;
        const price = data.apis?.pricing?.response?.summary?.total_price_usd || 'N/A';
        document.getElementById('modal-summary').innerHTML = `Client: ${clientName}<br>Resort: ${resortName}<br>Dates: ${dates}<br>Total Price: $${price}`;
        document.getElementById('confirmationModal').style.display = 'block';
    }

    document.getElementById('confirm-reservation').addEventListener('click', showConfirmationModal);
    document.querySelector('.close-button').addEventListener('click', () => document.getElementById('confirmationModal').style.display = 'none');
    document.getElementById('modal-cancel-btn').addEventListener('click', () => document.getElementById('confirmationModal').style.display = 'none');

    document.getElementById('modal-confirm-btn').addEventListener('click', () => {
        document.getElementById('confirmationModal').style.display = 'none';
        // Here you might want to add logic to reset the state or fetch the next client
        showNoClientsView();
    });

    fetchHotelData().then(data => {
        hotelDataStore = data;
        renderClientList();
        loadScenario('familia'); // Load a default scenario
    });
});
