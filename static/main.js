$(document).ready(function () {
    $("#chatForm").on("submit", function (event) {
        event.preventDefault(); // Explicitly prevent form submission

        var rawText = $("#text").val();

        // Basic input validation
        if (rawText.trim() === "") {
            // Optionally show an alert or provide user feedback for empty input
            return;
        }

        var userHtml = '<p class="userText"><span>' + rawText + "</span><img id='user_img' src='/static/user.png'/></p>";

        $("#text").val("");
        $("#chatbox").append(userHtml);
        document.getElementById("userInput").scrollIntoView({
            block: "start",
            behavior: "smooth",
        });

        $.ajax({
            data: { msg: rawText },
            type: "POST",
            url: "/get",
        })
            .done(function (data) {
                var botHtml = '<p class="botText"><img id="bot-img" src="/static/bot.png"/><span>' + data + "</span></p>";
                $("#chatbox").append($.parseHTML(botHtml));
                document.getElementById("userInput").scrollIntoView({
                    block: "start",
                    behavior: "smooth",
                });

                // Auto-scroll to the bottom after appending both user and bot messages
                setTimeout(function () {
                    $("#chatbox").animate({ scrollTop: $("#chatbox")[0].scrollHeight }, "slow");
                }, 100);
            })
            .fail(function (jqXHR, textStatus, errorThrown) {
                console.error("AJAX request failed:", textStatus, errorThrown);
                // Optionally show an alert or provide user feedback for the failed request
            });
    });
});
