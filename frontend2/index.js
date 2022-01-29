
import axios;
Vue.prototype.$http = axios;
var app = Vue.createApp({
    data() {
        return {
            posts: [],
        };
    },

    methods: {
        async getData() {
            try {
                const response = await this.$http.get(
                    "http://localhost:8000/ping"
                );
                this.posts = response.data;
            } catch (error) {
                console.log(error);
            }
        },
    },

    created() {
        this.getData();
    },
})

app.mount('#app')